import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm


# Define U-Net model for dot detection
class UNet(nn.Module):      #class initialization
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._encoder_block(1, 32)  # Input is grayscale (1 channel)
        self.enc2 = self._encoder_block(32, 64)
        self.enc3 = self._encoder_block(64, 128)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling)
        self.dec3 = self._decoder_block(256 + 128, 128)
        self.dec2 = self._decoder_block(128 + 64, 64)
        self.dec1 = self._decoder_block(64 + 32, 32)
        
        # Final layer (outputs heatmap)
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3_out)
        
        # Decoder with skip connections
        dec3_out = self.dec3(torch.cat((bottleneck, enc3_out), dim=1))
        dec2_out = self.dec2(torch.cat((dec3_out, enc2_out), dim=1))
        dec1_out = self.dec1(torch.cat((dec2_out, enc1_out), dim=1))
        
        # Final output
        heatmap = self.sigmoid(self.final(dec1_out))
        return heatmap
