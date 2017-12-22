import numpy as np
import matplotlib
matplotlib.use("Agg")
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

import matplotlib.pyplot as plt
import itertools

from preprocess import Rescale
from preprocess import RandomCrop
from preprocess import ToTensor
from preprocess import Normalize


par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Model params
useGPU=True
model_d = 128
color_d = 1

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20000
print_interval = 100

# Single Chinese character dataset
transformed_dataset = ChineseCharacterDataset(
                root_dir=par_path + '/images/',
                transform=transforms.Compose([
                    Rescale(64),
                    ToTensor(color_d),
                    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
    #plt.show()

def save_result(G, num_epoch, path, useGPU=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    if useGPU:
        z_ = z_.cuda()
    z_ = Variable(z_, volatile=True)

    G.eval()
    test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()

def save_train_hist(hist, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()

# def get_generator_input_sampler():
#     # return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian
#     Z = np.random.random((500,500))   # Test data
#     plt.imshow(Z, cmap='gray', interpolation='nearest')
#     plt.show()
#     return Z

# network
G = generator(model_d, color_d)
D = discriminator(model_d, color_d)

if useGPU:
    G=G.cuda()
    D=D.cuda()
    
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
if useGPU:
    BCE_loss = BCE_loss.cuda()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

print('training start!')
start_time = time.time()
D_losses = []
G_losses = []
epoch_start_time = time.time()

for epoch in range(train_epoch):

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

        if useGPU:
            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        else:
            x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)

        D_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        if useGPU:
            z_ = z_.cuda()
        z_ = Variable(z_)
        G_result = G(z_)

        D_result = D(G_result.detach()).squeeze()
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
        if useGPU:
            z_=z_.cuda()
        z_ = Variable(z_)

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    if (epoch+1)%print_interval != 0:
        continue

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))


    p = par_path + '/results/CHINESE_CHAR_DCGAN_' + str(epoch + 1) + '.png'
    if not os.path.exists(par_path + '/results/'):
        os.makedirs(par_path + '/results/')
    save_result(G, (epoch+1), path=p, useGPU=useGPU)

    save_train_hist(train_hist, path=par_path + '/results/CC_DCGAN_train_hist.png')

    
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
