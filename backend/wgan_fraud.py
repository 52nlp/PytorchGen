# Author: Yu-Hsuan Chen (Albert)
# Train W-GAN Model
import argparse
import datetime
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from data_treatment import DataSet, DataAtts  # YH
import pandas as pd
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data_set", type=str, default='fraud', help="data_set for training")
parser.add_argument("--n_epochs", type=int, default=1500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=30, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)
data_dic = {"fraud": "creditcard_1_train_no_label.csv"}
img_shape = (opt.channels, opt.img_size, 1)

cuda = True if torch.cuda.is_available() else False


def noise(quantity):
    noise_z = Variable(Tensor(np.random.normal(0, 1, (quantity, opt.latent_dim))))
    return noise_z


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True): # change from False to True 2019/9/8
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 256, normalize=True),
            *block(256, 512),
            nn.Linear(512, int(np.prod(img_shape))),
            # nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # print (img.shape[0], img_shape)
        img = img.view(img.shape[0], *img_shape)
        # print('shape:', img.shape)
        # print(img)
        return img

    def create_data(self, quantity):
        points = noise(quantity)
        try:
            data = self.forward(points.cuda())
        except RuntimeError:
            data = self.forward(points.cpu())
        return data.detach().cpu().numpy()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader


database = DataSet(csv_file=f"./original_data/{data_dic[opt.data_set]}", root_dir="./")

dataloader = torch.utils.data.DataLoader(database, batch_size=opt.batch_size, shuffle=True)
num_batches = len(dataloader)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------


batches_done = 0
for epoch in range(opt.n_epochs):

    for i, imgs in enumerate(dataloader):
        # print(i, imgs.shape)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1
torch.save({
    'epoch': epoch,

    'model_state_dict': generator.state_dict(),
    'optimizer_state_dict': optimizer_G.state_dict(),

}, f"models/{opt.data_set}/generator_wgan.pt")

torch.save({
    'epoch': epoch,

    'model_state_dict': discriminator.state_dict(),
    'optimizer_state_dict': optimizer_D.state_dict(),

}, f"models/{opt.data_set}/discriminator_wgan.pt")

# to generate fake data to train classifier


def gen_fake_data(n=20000):
    print("generating data...")
    device = torch.device('cpu')
    model = Generator()
    checkpoint = torch.load(f"models/{opt.data_set}/generator_wgan.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model_epoch = checkpoint['epoch']



    generated_data_points = model.create_data(n)

    # to save generated results
    original_db_name = opt.data_set
    now = datetime.datetime.now()
    name = f"wgan_{now.strftime('%Y-%m-%d')}_{str(n)}"
    generated_data_points = np.squeeze(generated_data_points)
    print(generated_data_points.shape)
    df_orig = pd.read_csv(f"./original_data/{data_dic[opt.data_set]}")
    print(generated_data_points, generated_data_points.shape)

    df = pd.DataFrame(generated_data_points, columns=df_orig.columns)
    df['Class'] = 1
    print(df.head())
    df.to_csv(f"fake_data/{original_db_name}/{name}.csv", index=False)

# gen_fake_data()

