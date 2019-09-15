# -*- coding: utf-8 -*-
# Author: Yu-Hsuan Chen (Albert)
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
latent_dim = 100
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def noise(quantity):
    noise_z = Variable(Tensor(np.random.normal(0, 1, (quantity, latent_dim))))
    return noise_z


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 256),
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


num_columns = 30
img_shape = (1, num_columns, 1)

data_set = "fraud"
data_dic = {"fraud": "creditcard_1_train_no_label.csv"}


def gen_fake_data(n=20000):
    # this function will generate n samples
    print("generating data...")
    device = torch.device('cpu')
    model = Generator()

    checkpoint = torch.load(f"models/{data_set}/generator_wgan.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_epoch = checkpoint['epoch']
    model.eval()
    generated_data_points = model.create_data(n)

    # to save generated results

    now = datetime.datetime.now()
    name = f"wgan_{now.strftime('%Y-%m-%d')}_{str(n)}"
    generated_data_points = np.squeeze(generated_data_points)

    df_orig = pd.read_csv(f"./original_data/{data_dic[data_set]}")
    print("data shape:", generated_data_points.shape)

    df = pd.DataFrame(generated_data_points, columns=df_orig.columns)
    df['Class'] = 1
    print(df.head())
    df.to_csv(f"fake_data/{data_set}/{name}.csv", index=False)
    print("Output file:", f"fake_data/{data_set}/{name}.csv")


#gen_fake_data(30000)
