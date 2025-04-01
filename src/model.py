import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from collections import OrderedDict

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, choice=1):
        super(BasicConvBlock, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3),  #Dropout added after ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    '''For an input of 3 channels'''
    def __init__(self, block_type, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 16

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)

        self.block1 = self.build_layer(block_type, 16, num_blocks[0], starting_stride=1)
        self.block2 = self.build_layer(block_type, 32, num_blocks[1], starting_stride=2)
        self.block3 = self.build_layer(block_type, 64, num_blocks[2], starting_stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)  # Dropout before final FC layer
        self.linear = nn.Linear(64, 2)  # 2 classes (Lens vs Non-Lens)

    def build_layer(self, block_type, out_channels, num_blocks, starting_stride):
        stride_list = [starting_stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in stride_list:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out) 
        out = self.linear(out)
        return out


def ResNet26():
    return ResNet(block_type=BasicConvBlock, num_blocks=[4, 4, 4])  # 6k+2 layers = 26 

