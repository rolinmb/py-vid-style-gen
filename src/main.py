from lib import *
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

def train(generator, discriminator, dataloader, epochs, device):
    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            #TODO: Training here
            pass
        print('lib.py : train() :: Training epoch {epoch+1}/{epochs} finished')



if __name__ == '__main__':
    pass
