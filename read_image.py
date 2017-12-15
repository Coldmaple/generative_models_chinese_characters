### Transfer binary image data to numpy array 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import math
import os
import os.path


def fread(fid, nelements, dtype):
    if dtype is np.str:
        dt = np.uint8 
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array

def read_image(fid):
    
    binary_num = fread(fid, 1, np.int32)[0][0]
    character_num = (int)(binary_num)
    binary_height = fread(fid, 1, np.uint8)[0, 0]
    height = (int)(binary_height)

    size = height*height

    image_mat = fread(fid, size*character_num, np.uint8)[:,0]

    image_dat = np.zeros((character_num, size))
    for i in range(character_num):
        image_dat[i] = image_mat[size*i:size*(i+1)]

    images = image_dat.reshape([character_num, height, height])
    # a single file is converted to numpy array: [6825, 128, 128]

    plt.imshow(images[0])
    plt.show()

    return images

def read_all_images():
    characters = np.zeros([6825, 122, 128, 128])
    par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    char_path = os.path.abspath(os.path.join(par_path, os.pardir)) + '/character_images/'
    for filename in os.listdir(char_path):
        fpath = char_path + filename
        print(fpath)
        file_id = open(fpath, 'rb')
        image = read_image(file_id)
        for i in range(0, 6825):
            # print(image[i])
            print(image[i].shape)
            plt.imshow(image[i])
            plt.show()
            # characters[i] = image[i]
        print(characters)

read_all_images()
