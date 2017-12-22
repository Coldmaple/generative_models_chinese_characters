import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ChineseCharacterDataset(Dataset):
    """Chinese characters dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        list = os.listdir(self.root_dir)
        number_files = len(list)
        return number_files

    def __getitem__(self, idx):
        img_name = self.root_dir + '/' + str(idx) + '.jpg'
        image = io.imread(img_name)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample
        


# def read_all_images():

#     characters = [6825, 122, 128, 128]
#     par_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#     char_path = os.path.abspath(os.path.join(par_path, os.pardir)) + '/Gan_chinese_characters/character/'

#     file_index = 0
#     for filename in os.listdir(char_path):
#         file_index += 1
#         fpath = char_path + filename
#         print(fpath)
#         file_id = open(fpath, 'rb')
#         image = read_image(file_id)
#         for i in range(0, len(image)):
#             characters[i][file_index - 1] = image[i]

#     print(characters)
