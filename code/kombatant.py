import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Kombatant(nn.Module):
    def __init__(self, 
                 num_conv_layers,
                 conv_channel_sizes,
                 image_size,
                additional_input_size, output_size):
        
        super(Kombatant, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_channel_sizes = conv_channel_sizes
        self.additional_input_size = additional_input_size
        self.output_size = output_size

        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels=conv_channel_sizes[i], out_channels=conv_channel_sizes[i+1], kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # add a 1x1 convolutional layer
        conv_layers.append(nn.Conv2d(in_channels=conv_channel_sizes[-1], out_channels= 1, kernel_size=1, stride=1, padding=0))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.fc_layers = nn.Sequential(nn.Linear( image_size*image_size*conv_channel_sizes[-1] + additional_input_size, 512),
        

