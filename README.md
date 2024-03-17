# Training-Free Generator for Few-Shot Class Incremental Learning

PyTorch implementation of Training-Free Generator for Few-Shot Class Incremental Learning

## Abstract

In Class Incremental Learning (CIL), generative replay methods play a crucial role by training a generator
to recreate old session samples when new sessions are introduced. However, in the more realistic and stringent
scenario of Few-Shot Class Incremental Learning (FSCIL), limited data in new sessions often leads to overfitting
of the generator, resulting in poor generation and classification performance. To address this challenge, 
we introduce the concept of zero-shot learning and propose a novel conditional generator capable of directly
generating samples for new sessions without explicit training. Our approach involves training a feature 
extractor to reduce the training complexity of the generator and then leveraging a semantic feature 
conditional generator, which combines Variation AutoEncoder (VAE) and Generative Adversarial Network (GAN) 
architectures, to generate image features based on semantic features. By utilizing semantic features from 
both old and new classes, our method generates corresponding pseudo image features for new sessions, which 
are then combined with real image features for classifier fine-tuning. Extensive experiments conducted on 
the miniImagenet, CUB200, and CIFAR100 datasets demonstrate that our proposed method achieves state-of-the-art 
performance in FSCIL tasks. Our work addresses the limitations of existing methods and opens avenues for 
further research in Few-Shot Class Incremental Learning. 

## Requirements

tqdm~=4.64.1
numpy~=1.23.5
pillow~=9.3.0
torchvision~=0.15.0

## Datasets

We follow [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training.  
For CIFAR100, the dataset will be download automatically.  
For miniImagenet and CUB200, you can download
from [here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing). Please put the
downloaded file under `data/` folder and unzip it:

    $ tar -xvf miniimagenet.tar 
    $ tar -xvzf CUB_200_2011.tgz

## Training scripts

cifar100

    $ python main.py -dataset cifar100 -epochs_base 150 -lr_base 0.1 -batch_size_base 128 -dataroot YOUR_DATA_PATH

mini_imagenet

    $ python main.py -dataset mini_imagenet -epochs_base 150 -lr_base 0.1 -batch_size_base 128 -dataroot YOUR_DATA_PATH

cub200

    $ python main.py -dataset cub200 -epochs_base 150 -lr_base 0.01 -batch_size_base 128 -dataroot YOUR_DATA_PATH

## Notice

Different machines may yield different results; therefore, we provide the trained model parameters for users 
to check the results of our work. The trained model parameters are located in the `checkpoint` directory. 
To use them, set the base learning rate to 0.0001 and run the model.

## Acknowledgment

Our project references the codes in the following repos.

- [fscil](https://github.com/xyutao/fscil)
- [DeepEMD](https://github.com/icoz69/DeepEMD)
