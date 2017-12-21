import numpy as np
import matplotlib.pyplot as plt

import os, time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils

from data import ChineseCharacterDataset
from model import generator, discriminator

import util
from preprocess import Rescale
from preprocess import RandomCrop
from preprocess import ToTensor

from picutil import show_result

# Model params
g_input_size = 100     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 1    # size of generated output vector
d_input_size = 100   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 30000
print_interval = 200
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

# training parameters
batch_size = 16
lr = 0.0002
train_epoch = 20

# Single Chinese character dataset
transformed_dataset = ChineseCharacterDataset(
                root_dir='/media/sf_sharewithvm/cv/generative_models_chinese_characters/image/3103',
                transform=transforms.Compose([
                    Rescale(64),
                    ToTensor()
                ]))

dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

# Helper function to show a batch (This is for test)
def show_batch_image(sample_batched):
    images_batch = \
            sample_batched['image']

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size())
    #show_batch_image(sample_batched)
    plt.show()

# def get_generator_input_sampler():
#     # return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian
#     Z = np.random.random((500,500))   # Test data
#     plt.imshow(Z, cmap='gray', interpolation='nearest')
#     plt.show()
#     return Z

# network
G = generator(2)
D = discriminator(2)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    epoch_start_time = time.time()

    for dic in dataloader:
        x_ = dic['image']
        # Convert ByteTensor to FloatTensor
        x_ = x_.type(torch.FloatTensor)
        #print(x_)
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)
        D_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_)
        G_result = G(z_)

        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # D_losses.append(D_train_loss.data[0])
        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_)

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))


    p = 'Random_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
    show_result(G, (epoch+1), save=True, path=p, isFix=False)
