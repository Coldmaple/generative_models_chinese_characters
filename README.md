# generative_models_chinese_characters

## Development Environment

* Python 3.6.2
* pytorch 0.3.0
* torchvision 0.2.0
* matplotlib 2.0.2
* imageio 2.2.0
* scipy 1.0.0

## Data
We use open source data from http://www.iapr-tc11.org/mediawiki/index.php/Harbin_Institute_of_Technology_Opening_Recognition_Corpus_for_Chinese_Characters_(HIT-OR3C)
Download the data from the website above, choose 'Offline characters' as your database. Unzip this file, you will find all the data is stored in binary version. There are 6825 Chinese characters in total. These handwritten characters are collected from 122 individuals. Each image is of 128*128. Detailed instructions of how to use this data can be found in the website.

We made small changes on loading this data. Since we want to train how to write Chinese characters, we put single character (i.e 大） of all sampling writers into one file.

## Model
We use DCGAN as our model, a simple example of MNIST can be found here:
https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_MNIST_DCGAN.py


