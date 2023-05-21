"""Test a model and generate submission CSV.

> python3 train.py --conf ../cfg/s1.yml 

Usage:
    > python train.py --load_path PATH --name NAME
    where
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the train run
"""
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchfile
from PIL import Image
from sklearn import metrics, model_selection
from torch.autograd import Variable

import args
# import config
import dataset
import engine
import util
import wandb
from layers import (Stage1Discriminator, Stage1Generator, Stage2Discriminator,
                    Stage2Generator)
# from engine import fid
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
data_args = args.get_all_args()

print("__"*80)
print("Imports Done...")

checkpoint = False
CHECKPOINT_PATH = '../output/model/stage1'
CHECKPOINT_GEN_PATH = CHECKPOINT_PATH + '/netG.pth'
CHECKPOINT_DIS_PATH = CHECKPOINT_PATH + '/netD.pth'

# run = wandb.init(project='StackGAN-RoBERTa', name='stage1', id='qlium3kd', notes='This is training stage 1', resume=True,
#                  tags=['stage1', 'roberta'], dir=data_args.log_dir)
run = wandb.init(project='StackGAN-RoBERTa', name='stage1-roberta', notes='This is training stage 1', resume=True,
                 tags=['stage1', 'roberta'], dir=data_args.log_dir)


def fetch_checkpoints(gen, disc):
    if wandb.run.resumed:
        print("________ FETCHING CHECKPOINTS ________")
        checkpoint_gen = torch.load(CHECKPOINT_GEN_PATH)
        checkpoint_disc = torch.load(CHECKPOINT_DIS_PATH)

        gen.load_state_dict(checkpoint_gen['model_state_dict'])
        print("__"*80)
        print("Generator loaded from: ", CHECKPOINT_GEN_PATH)
        print("__"*80)
        disc.load_state_dict(checkpoint_disc['model_state_dict'])
        print("__"*80)
        print("Discriminator loaded from: ", CHECKPOINT_DIS_PATH)
        print("__"*80)

        epoch_gen = checkpoint_gen['epoch'] + 1
        print(f'----- previous epoch = {epoch_gen - 1} -----')
        epoch_disc = checkpoint_disc['epoch']
        loss_metrics_gen = checkpoint_gen['loss_metrics']
        print(f'----- previous loss_metrics = {loss_metrics_gen} -----')
        loss_metrics_disc = checkpoint_disc['loss_metrics']

        return gen, disc, epoch_gen


def load_stage1(args):
    # * Init models and weights:
    from layers import Stage1Discriminator, Stage1Generator
    if args.embedding_type == "roberta":
        netG = Stage1Generator(emb_dim=768)
        netD = Stage1Discriminator(emb_dim=768)
    else:
        netG = Stage1Generator(emb_dim=1024)
        netD = Stage1Discriminator(emb_dim=1024)

    netG.apply(engine.weights_init)
    netD.apply(engine.weights_init)

    # * Load saved checkpoints:
    # if args.NET_G_path != "" and args.NET_D_path != "":
    if len(os.listdir(CHECKPOINT_PATH)) > 0:
        netG, netD, epoch = fetch_checkpoints(netG, netD)
    else:
        epoch = 1

    # * Load on device:
    if args.device == "cuda":
        netG.cuda()
        netD.cuda()
    else:
        netG.cpu()
        netD.cpu()

    print("__"*80)
    print("GENERATOR:")
    print(netG)
    print("__"*80)
    print("DISCRIMINATOR:")
    print(netD)
    print("__"*80)

    return netG, netD, epoch


def load_stage2(args):
    # * Init models and weights:
    from layers import Stage1Generator, Stage2Discriminator, Stage2Generator
    if args.embedding_type == "roberta":
        Stage1_G = Stage1Generator(emb_dim=768)
        netG = Stage2Generator(Stage1_G, emb_dim=768)
        netD = Stage2Discriminator(emb_dim=768)
    else:
        Stage1_G = Stage1Generator(emb_dim=1024)
        netG = Stage2Generator(Stage1_G, emb_dim=1024)
        netD = Stage2Discriminator(emb_dim=1024)
    netG.apply(engine.weights_init)
    netD.apply(engine.weights_init)

    # * Load saved model:
    if len(os.listdir(CHECKPOINT_PATH)) > 0:
        netG, netD, epoch = fetch_checkpoints(netG, netD)
        print("Generator loaded from: ", CHECKPOINT_GEN_PATH)
        print("Discriminator loaded from: ", CHECKPOINT_DIS_PATH)
    elif len(os.listdir('../output/model')) > 0:
        stage1_gen_checkpoint = torch.load('../output/model/netG.pth')
        netG.stage1_gen.load_state_dict(
            stage1_gen_checkpoint['model_state_dict'])
        epoch = 1
        print("Generator 1 loaded from: '../output/model/netG.pth'")
    else:
        print("Please give the Stage 1 generator path")
        return

    # * Load on device:
    if args.device == "cuda":
        netG.cuda()
        netD.cuda()
    else:
        netG.cpu()
        netD.cpu()

    print("__"*80)
    print(netG)
    print("__"*80)
    print(netD)
    print("__"*80)

    return netG, netD, epoch


def run(args):

    if args.STAGE == 1:
        netG, netD, epoch = load_stage1(args)
    else:
        netG, netD, epoch = load_stage2(args)
    # Setting up device
    device = torch.device(args.device)

    # Load model
    netG.to(device)
    netD.to(device)

    nz = args.n_z
    batch_size = args.train_bs
    noise = Variable(torch.FloatTensor(batch_size, nz)).to(device)
    with torch.no_grad():
        fixed_noise = Variable(torch.FloatTensor(
            batch_size, nz).normal_(0, 1)).to(device)  # volatile=True
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1)).to(device)
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0)).to(device)

    gen_lr = args.TRAIN_GEN_LR
    disc_lr = args.TRAIN_DISC_LR

    lr_decay_step = args.TRAIN_LR_DECAY_EPOCH

    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=args.TRAIN_DISC_LR, betas=(0.5, 0.999))

    netG_para = []
    for p in netG.parameters():
        if p.requires_grad:
            netG_para.append(p)
    optimizerG = torch.optim.Adam(
        netG_para, lr=args.TRAIN_GEN_LR, betas=(0.5, 0.999))

    count = 0

    if args.embedding_type == "roberta":
        training_set = dataset.CUBDataset(pickl_file=args.train_filenames, img_dir=args.images_dir,
                                          roberta_emb=args.roberta_annotations_dir, stage=args.STAGE)
        testing_set = dataset.CUBDataset(pickl_file=args.test_filenames, img_dir=args.images_dir,
                                         roberta_emb=args.roberta_annotations_dir, stage=args.STAGE)
    else:
        training_set = dataset.CUBDataset(
            pickl_file=args.train_filenames, img_dir=args.images_dir, cnn_emb=args.cnn_annotations_emb_train, stage=args.STAGE)
        testing_set = dataset.CUBDataset(
            pickl_file=args.test_filenames, img_dir=args.images_dir, cnn_emb=args.cnn_annotations_emb_test, stage=args.STAGE)
    train_data_loader = torch.utils.data.DataLoader(
        training_set, batch_size=args.train_bs, num_workers=args.train_workers)
    test_data_loader = torch.utils.data.DataLoader(
        testing_set, batch_size=args.test_bs, num_workers=args.test_workers)
    # util.check_dataset(training_set)
    # util.check_dataset(testing_set)

    # best_accuracy = 0

    util.make_dir(args.image_save_dir)
    util.make_dir(args.model_dir)
    util.make_dir(args.log_dir)
    while epoch <= args.TRAIN_MAX_EPOCH:
        # for epoch in range(1, args.TRAIN_MAX_EPOCH+1):
        print("__"*80)
        start_t = time.time()

        if epoch % lr_decay_step == 0 and epoch > 0:
            gen_lr *= 0.5
            for param_group in optimizerG.param_groups:
                param_group["lr"] = gen_lr
            disc_lr *= 0.5
            for param_group in optimizerD.param_groups:
                param_group["lr"] = disc_lr

        errD, errD_real, errD_wrong, errD_fake, errG, kl_loss, count, loss_metrics = engine.train_new_fn(
            train_data_loader, args, netG, netD, real_labels, fake_labels,
            noise, fixed_noise,  optimizerD, optimizerG, epoch, count)

        wandb.log(loss_metrics, step=epoch)

        end_t = time.time()

        print(f"[{epoch}/{args.TRAIN_MAX_EPOCH}] Loss_D: {errD:.4f}, Loss_G: {errG:.4f}, Loss_KL: {kl_loss:.4f}, Loss_real: {errD_real:.4f}, Loss_wrong: {errD_wrong:.4f}, Loss_fake: {errD_fake:.4f}, Total Time: {end_t-start_t :.2f} sec")
        # args.TRAIN_SNAPSHOT_INTERVAL == 0
        # if epoch % 50 == 0 or epoch == 1:

        # if epoch % 50 == 0 or epoch == 1:
        util.save_model(netG, netD, epoch, loss_metrics, args)

        epoch += 1

    util.save_model(netG, netD, args.TRAIN_MAX_EPOCH, loss_metrics, args)


def sample(args, datapath):
    if args.STAGE == 1:
        netG, _, _ = load_stage1(args)
    else:
        netG, _, _ = load_stage2(args)
    netG.eval()

    # * Load text embeddings generated from the encoder:
    t_file = torchfile.load(datapath)
    captions_list = t_file.raw_txt
    embeddings = np.concatenate(t_file.fea_txt, axis=0)
    num_embeddings = len(captions_list)
    print(f"Successfully load sentences from: {args.datapath}")
    print(f"Total number of sentences: {num_embeddings}")
    print(f"Num embeddings: {num_embeddings} {embeddings.shape}")

    # * Path to save generated samples:
    save_dir = args.NET_G[:args.NET_G.find(".pth")]
    util.make_dir(save_dir)

    batch_size = np.minimum(num_embeddings, args.train_bs)
    nz = args.n_z
    noise = Variable(torch.FloatTensor(batch_size, nz))
    noise = noise.to(args.device)
    count = 0
    while count < num_embeddings:
        if count > 3000:
            break
        iend = count + batch_size
        if iend > num_embeddings:
            iend = num_embeddings
            count = num_embeddings - batch_size
        embeddings_batch = embeddings[count:iend]
        # captions_batch = captions_list[count:iend]
        text_embedding = Variable(torch.FloatTensor(embeddings_batch))
        text_embedding = text_embedding.to(args.device)

        # * Generate fake images:
        noise.data.normal_(0, 1)
        _, fake_imgs, mu, logvar = netG(text_embedding, noise)
        for i in range(batch_size):
            save_name = f"{save_dir}/{count+i}.png"
            im = fake_imgs[i].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            # print("im", im.shape)
            im = np.transpose(im, (1, 2, 0))
            # print("im", im.shape)
            im = Image.fromarray(im)
            im.save(save_name)
        count += batch_size


if __name__ == "__main__":
    args_ = args.get_all_args()
    args.print_args(args_)
    run(args_)
    # datapath = os.path.join(args_.datapath, "test/val_captions.t7")
    # sample(args_, datapath)
