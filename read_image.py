### Transfer binary image data to numpy array 

## import matplotlib.pyplot as plt
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

    image_mat = fread(fid, size*character_num, np.uint8)

    image_dat = np.zeros((character_num, size, 1))
    for i in range(character_num):
        image_dat[i] = image_mat[size*i:size*(i+1)]

    image = image_dat.reshape([character_num, height, height, 1])
    # a single file is converted to numpy array: [6825, 128, 128, 1]

    return image

    # plt.imshow(image[0])
    # plt.show()

def read_all_images():
    par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    char_path = os.path.abspath(os.path.join(par_path, os.pardir)) + '/character_images/'
    for filename in os.listdir(char_path):
        fpath = char_path + filename
        print(fpath)
        file_id = open(fpath, 'rb')
        image = read_image(file_id)
        print(image.shape)

read_all_images()
