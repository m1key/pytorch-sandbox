# PyThorch Sandbox

## Information

This is based on:
https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce

The corresponding Github repo is:
https://github.com/WillKoehrsen/pytorch_challenge

https://github.com/WillKoehrsen/pytorch_challenge/blob/master/Transfer%20Learning%20in%20PyTorch.ipynb

CUDA (Compute Unified Device Architecture) is "a parallel computing platform and application programming interface (API) model created by Nvidia".

## Performance

Dell XPS, Intel HD 5500: 5.5 hours to train and re-train.

My PC with NVidia 970: 9 minutes.

Google Colab: 20 minutes.

## Installation

### Linux (no CUDA)

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

### Windows (CUDA)

I installed Anaconda 5.3.1 on Windows 7.

Then, I installed CUDA 9.0 (because at this point in time PyTorch did not work with CUDA 10.0).
I did not install Visual Studio (even though it complained).

Then, I opened Anaconda Propmt from the Start menu.

> conda install pytorch -c pytorch

> pip install torchvision

> pip install torchsummary

### Google Colab

Firstly, it complained about the lack of pytorch. It offered to install it
automatically, and generated something like this:

> from os.path import exists

> from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag

> platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

> cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'

> accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

> !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision

> import torch

Then, I had to clone the repo to get the data. There must be a safer way:

> !git clone https://user:password@github.com/m1key/pytorch-sandbox.git

I got this error: module 'PIL.Image' has no attribute 'register_extensions'.

We need pillow version of 5.3.0. We will uninstall the older version first.

> !pip uninstall -y Pillow

Install the new one

> !pip install Pillow==5.3.0

Import the new one

> import PIL

> print(PIL.PILLOW_VERSION)

Menu > Runtime > Restart Runtime

Then, torchsummary had to be installed:

> !pip install torchsummary

I had to change the location of the files:

> traindir = 'pytorch-sandbox/datadir/train/'

> validdir = 'pytorch-sandbox/datadir/valid/'

> testdir = 'pytorch-sandbox/datadir/test/'

I had to upgrade seaborn.

> !pip uninstall -y seaborn

> !pip install seaborn==0.9.0

> import seaborn

> print(seaborn.__version__)

## Data

The images used for this were downloaded from http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download

I'm planning to save them in this repo to speed things up.
