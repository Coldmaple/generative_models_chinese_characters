import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pylab as pylab
from data import ChineseCharacterDataset
import read_image
import util

# Model params
g_input_size = 122     # Random noise dimension coming into generator, per output vector
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

def get_generator_input_sampler():
    # return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian
    Z = np.random.random((500,500))   # Test data
    plt.imshow(Z, cmap='gray', interpolation='nearest')
    plt.show()
    return Z

selected_char = util.getCharInd()


read_image.save_images(idxs=selected_char,
                       input_dir='/Gan_chinese_characters/character/',
                       image_dir='/Volumes/mhr2/Gan_chinese_characters/image')


# plt.imshow(characters[0][0])
# plt.show()

chinese_character_dataset = ChineseCharacterDataset(root_dir='/Volumes/mhr2/Gan_chinese_characters/character/')


