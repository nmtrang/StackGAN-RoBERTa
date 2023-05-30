"""Train and Eval functions

"""
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from ignite.metrics import FID, InceptionScore
from scipy import linalg
from scipy.linalg import sqrtm
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
from tqdm import tqdm

import args
import util

data_args = args.get_all_args()


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

    # real pairs
    real_logits = disc(cond, real_imgs)
    errD_real = loss_fn(real_logits, real_labels)

    # wrong pairs
    wrong_logits = disc(cond[1:], fake_imgs[:(batch_size-1)])
    errD_wrong = loss_fn(wrong_logits, fake_labels[1:])

    # fake pairs
    fake_logits = disc(cond, fake_imgs)
    errD_fake = loss_fn(fake_logits, fake_labels)

    errD = errD_real + (errD_fake + errD_wrong) * 0.5

    # return errD, errD_real.data[0], errD_wrong.data[0], errD_fake.data[0]

    # real_loss = loss_fn(disc(cond, real_imgs), real_labels)
    # fake_loss = loss_fn(disc(cond, fake_imgs), fake_labels)
    # wrong_loss = loss_fn(disc(cond[1:], real_imgs[:-1]), fake_labels[1:])
    # loss = real_loss + (fake_loss + wrong_loss) * 0.5

    return errD, errD_real, errD_wrong, errD_fake


def gen_loss(disc, fake_imgs, real_labels, conditional_vector):
    loss_fn = nn.BCELoss()
    cond = conditional_vector.detach()
    # fake pairs
    fake_logits = disc(cond, fake_imgs)
    errD_fake = loss_fn(fake_logits, real_labels)

    return errD_fake
    # fake_loss = loss_fn(disc(cond, fake_imgs), real_labels)
    # return fake_loss


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
    real_images_all, fake_images_all = [], []
    for batch_id, data in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Train Epoch {epoch}/{args.TRAIN_MAX_EPOCH}",
    ):
        # print(f'Loading {batch_id}: {data}')
        # * Prepare training data:
        text_emb, real_images = data
        text_emb = text_emb.to(args.device)
        real_images = real_images.to(args.device)

        # * Generate fake images:
        noise.data.normal_(0, 1)
        _, fake_images, mu, logvar = netG(text_emb, noise)

        # * Resize real & fake images to 2x2 kernel size:
        real_images_resized = F.interpolate(real_images, size=(
            299, 299), mode='bilinear', align_corners=True)
        fake_images_resized = F.interpolate(fake_images, size=(
            299, 299), mode='bilinear', align_corners=True)

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

        if batch_id % 100 == 0:
            fid_score = 0
            is_score = 0
            try:
                print('Calculating FID and IS...')
                fid_score, is_score = calculate_fid_and_is(
                    real_images_resized, fake_images_resized)
                print('Finally!!! Succeed calculating FID and IS...')
            except Exception as e:
                print(e)
                print('Failed to calculate FID and IS...')
            print(
                f"[Stage 1] Epoch {epoch}: FID={fid_score:.2f}, IS_score={is_score:.2f}")
            loss_metrics = {"D_loss": errD.data,
                            "D_loss_real": errD_real.data,
                            "D_loss_wrong": errD_wrong.data,
                            "D_loss_fake": errD_fake.data,
                            "G_loss": errG.data,
                            "KL_loss": kl_loss.data,
                            "Inception_score": is_score,
                            "FID": fid_score}

            # * save the image result for each epoch:
            # lr_fake, fake, _, _ = nn.parallel.data_parallel(netG, inputs, device_ids=[0])
            lr_fake, fake, _, _ = netG(text_emb, fixed_noise)

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

    return errD_, errD_real_, errD_wrong_, errD_fake_, errG_, kl_loss_, count, loss_metrics


def eval_fn(data_loader, model, device, epoch):
    model.eval()
    fin_y = []
    fin_outputs = []
    LOSS = 0.

    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            text_embs, images = data

            # Loading it to device
            text_embs = text_embs.to(device, dtype=torch.float)
            images = images.to(device, dtype=torch.float)

            # getting outputs from model and calculating loss
            outputs = model(text_embs, images)
            loss = loss_fn(outputs, images)  # TODO figure this out
            LOSS += loss

            # for calculating accuracy and other metrics # TODO figure this out
            fin_y.extend(images.view(-1, 1).cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(
                outputs).cpu().detach().numpy().tolist())

    LOSS /= len(data_loader)
    return fin_outputs, fin_y, LOSS


def calculate_fid_and_is(real_images, fake_images, num_images=1000, z_dim=100, device=torch.device('cuda')):
    # model = inception_v3(pretrained=True, transform_input=False)
    # model.eval()
    # define fid and is metric
    fid_metric = FID(device=device)
    is_metric = InceptionScore(device=device)

    fid_metric.update((real_images, fake_images))
    is_metric.update(fake_images)
    # calculate FID and IS between generated images and real images
    fid_score = fid_metric.compute()
    is_score = is_metric.compute()

    return fid_score, is_score
