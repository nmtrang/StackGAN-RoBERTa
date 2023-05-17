"""Train and Eval functions

"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from pytorch_fid import fid_score
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

import args
import util

data_args = args.get_all_args()
wandb.init(project='StackGAN-RoBERTa', notes='This is training stage 1',
           tags=['stage1', 'roberta'], dir=data_args.log_dir, resume=True)


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def disc_loss(disc, real_imgs, fake_imgs, real_labels, fake_labels, conditional_vector):
    loss_fn = nn.BCELoss()
    batch_size = real_imgs.shape[0]
    cond = conditional_vector.detach()
    fake_imgs = fake_imgs.detach()

    real_loss = loss_fn(disc(cond, real_imgs), real_labels)
    fake_loss = loss_fn(disc(cond, fake_imgs), fake_labels)
    wrong_loss = loss_fn(disc(cond[1:], real_imgs[:-1]), fake_labels[1:])
    loss = real_loss + (fake_loss + wrong_loss) * 0.5
    return loss, real_loss, wrong_loss, fake_loss


def gen_loss(disc, fake_imgs, real_labels, conditional_vector):
    loss_fn = nn.BCELoss()
    cond = conditional_vector.detach()
    fake_loss = loss_fn(disc(cond, fake_imgs), real_labels)
    return fake_loss


def extract_features(images):
    """
    Extracts features from images using the Inception model.
    """
    # Normalize the images to the expected range for the Inception model
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    images = torchvision.transforms.Resize((299, 299))(images)

    # Pass the images through the Inception model
    with torch.no_grad():
        features = inception_model(images).detach().cpu().numpy()

    return features


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def train_new_fn(
    data_loader,
    args,
    netG,
    netD,
    real_labels,
    fake_labels,
    noise,
    fixed_noise,
    optimizerD,
    optimizerG,
    epoch,
    count
):
    errD_, errD_real_, errD_wrong_, errD_fake_, errG_, kl_loss_ = 0, 0, 0, 0, 0, 0
    wandb.watch(netG)
    wandb.watch(netD)
    real_images_all, fake_images_all = [], []
    for batch_id, data in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Train Epoch {epoch}/{args.TRAIN_MAX_EPOCH}",
    ):
        # * Prepare training data:
        text_emb, real_images = data
        text_emb = text_emb.to(args.device)
        real_images = real_images.to(args.device)

        # * Generate fake images:
        noise.data.normal_(0, 1)
        _, fake_images, mu, logvar = netG(text_emb, noise)

        # * Update D network:
        netD.zero_grad()
        errD, errD_real, errD_wrong, errD_fake = disc_loss(
            netD, real_images, fake_images, real_labels, fake_labels, text_emb
        )
        errD.backward()
        optimizerD.step()

        # * Update G network:
        netG.zero_grad()
        errG = gen_loss(netD, fake_images, real_labels, text_emb)
        kl_loss = KL_loss(mu, logvar)
        errG_total = errG + kl_loss * args.TRAIN_COEFF_KL
        errG_total.backward()
        optimizerG.step()

        count += 1

        if batch_id % 10 == 0:
            loss_metrics = {"D_loss": errD.data,
                            "D_loss_real": errD_real.data,
                            "D_loss_wrong": errD_wrong.data,
                            "D_loss_fake": errD_fake.data,
                            "G_loss": errG.data,
                            "KL_loss": kl_loss.data}

            wandb.log(loss_metrics, step=count)

            # * save the image result for each epoch:
            lr_fake, fake, _, _ = netG(text_emb, fixed_noise)
            try:
                fid = calculate_fid(real_images, fake, args)
                wandb.log({"FID": fid})
            except:
                pass

            try:
                inception_score = calculate_inception_score(
                    netG, args.IS_NUM_SAMPLES, args.BATCH_SIZE, args.device, kl_loss.data)
                wandb.log({"Inception Score": inception_score})
            except:
                pass

            util.save_img_results(real_images, fake, epoch, args)
            if lr_fake is not None:
                util.save_img_results(None, lr_fake, epoch, args)

            errD_ += errD
            errD_real_ += errD_real
            errD_wrong_ += errD_wrong
            errD_fake_ += errD_fake
            errG_ += errG
            kl_loss_ += kl_loss

    errD_ /= len(data_loader)
    errD_real_ /= len(data_loader)
    errD_wrong_ /= len(data_loader)
    errD_fake_ /= len(data_loader)
    errG_ /= len(data_loader)
    kl_loss_ /= len(data_loader)

    return errD_, errD_real_, errD_wrong_, errD_fake_, errG_, kl_loss_, count


def calculate_fid(real_images, fake_images, args):
    # Resize images to 2048x2048 if necessary
    if args.image_size < 2048:
        real_images = torch.nn.functional.interpolate(
            real_images, size=(2048, 2048), mode='bilinear', align_corners=False)
        fake_images = torch.nn.functional.interpolate(
            fake_images, size=(2048, 2048), mode='bilinear', align_corners=False)

    # Calculate FID score using pytorch-fid
    fid = fid_score.calculate_fid_given_tensors(
        real_images, fake_images, args.device, args.FID_BATCH_SIZE)

    return fid


def calculate_inception_score(netG, num_samples, batch_size, device, kl_loss=None):
    """Calculate the Inception Score for a generator network.

    Args:
        netG (nn.Module): The generator network.
        num_samples (int): The number of fake images to generate.
        batch_size (int): The batch size to use for generating the images.
        device (str): The device to use for computation.
        kl_loss (float, optional): The pre-computed KL loss. Defaults to None.

    Returns:
        float: The Inception Score.
    """
    netG.eval()
    inception_model = inception_v3(
        pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    if kl_loss is None:
        # * Generate latent vectors:
        z = torch.randn(num_samples, netG.z_dim, device=device)

        # * Generate fake images:
        fake_images = []
        for i in range(0, num_samples, batch_size):
            with torch.no_grad():
                batch_z = z[i:i+batch_size]
                batch_images = netG.generate_from_z(batch_z).cpu()
            fake_images.append(batch_images)
        fake_images = torch.cat(fake_images, dim=0)

        # * Calculate KL divergence:
        mu, logvar = netG.encode(fake_images)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # * Generate fake images:
    fake_images = []
    for i in range(0, num_samples, batch_size):
        with torch.no_grad():
            batch_z = z[i:i+batch_size]
            batch_images = netG.generate_from_z(batch_z).cpu()
        fake_images.append(batch_images)
    fake_images = torch.cat(fake_images, dim=0)

    # * Compute Inception Score:
    fake_scores = inception_model(fake_images)[0]
    scores = F.softmax(fake_scores, dim=1).data.cpu().numpy()
    scores = np.mean(scores, axis=0)
    kl_score = np.exp(kl_loss.data.cpu().numpy())

    return np.sum(scores * np.log(scores / kl_score))
