# PyThorch Sandbox

## Information

This is based on:
https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce

The corresponding Github repo is:
https://github.com/WillKoehrsen/pytorch_challenge

https://github.com/WillKoehrsen/pytorch_challenge/blob/master/Transfer%20Learning%20in%20PyTorch.ipynb

CUDA (Compute Unified Device Architecture) is "a parallel computing platform and application programming interface (API) model created by Nvidia".

## Installation

I installed Anaconda 5.3.1 on Fedora Linux.

Then, I installed PyTorch:

> conda install pytorch-cpu torchvision-cpu -c pytorch

I don't know why this required pip install:

> pip install torchsummary

### CUDA

On my work laptop, I found out the graphics card using this command.

> lspci -v | less

(Look for VGA.)

It is Intel Corporation HD Graphics 5500. It does not support CUDA.


## Data

The images used for this were downloaded from http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download

I'm planning to save them in this repo to speed things up.