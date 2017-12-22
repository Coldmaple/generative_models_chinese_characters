### Transfer binary image data to numpy array 

import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import scipy.misc

def getCharByte(label_path):
    fpath = label_path + "/chars.txt"
    out = []
    with open(fpath,'rb') as f:
        while True:
            c = f.readline()
            if not c:
              break
            out = out+c.split()
    return out

def getCharByteNospace(label_path):
    fpath = label_path + "/chars.txt"
    out = []
    with open(fpath,'rb') as f:
        while True:
            c = f.read(2)
            if not c:
                break
            out.append(c)
    return out

def getCharInd(Space=True):
    par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    label_path = par_path+ '/character_images_labels/'
    
    if Space:
        chars = getCharByte(label_path)
    else:
        chars = getCharByteNospace(label_path)

    fpath = label_path + "/labels.txt"
    i = 0
    out = []
    with open(fpath,'rb') as f:
        while True:
            c = f.read(2)
            if not c:
                break
            if c in chars:
                out.append(i)
            i += 1
    return out

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

    # plt.imshow(images[0])
    # plt.show()

    return images

def save_images(idxs, input_dir, image_dir):

    par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    char_path = par_path + input_dir
    image_dir = par_path + image_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    file_index = 0
    for filename in os.listdir(char_path):
        fpath = char_path + filename
        print(fpath)
        file_id = open(fpath, 'rb')
        image = read_image(file_id)
        for i in range(0, len(idxs)):
            basename = idxs[i]
            folder = image_dir
            outfile = '%s/%s.jpg' % (folder, str(file_index))
            scipy.misc.imsave(outfile, image[idxs[i]])
        
            file_index += 1

if __name__ == "__main__":

    selected_char = getCharInd(Space=False)
    
    # Save images to disk
    save_images(idxs=selected_char,
            input_dir='/character_images/',
            image_dir='/images/')

