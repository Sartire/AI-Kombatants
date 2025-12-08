import torch
import torch.nn as nn


import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule

# Define your custom env class by subclassing `TorchRLModule`:
class Kombatant(TorchRLModule):
    def setup(self):
        # You have access here to the following already set attributes:
        # self.observation_space
        # self.action_space
        # self.inference_only
        # self.model_config  # <- a dict with custom settings
        self.input_shape = self.observation_space.shape
        hidden_dim = self.model_config["hidden_dim"]
        output_dim = self.action_space.n
        
        self.conv_layers_spec = self.model_config['conv_layers_spec']
        self.additional_input_size = self.model_config['additional_input_size']
        
        # Define and assign your torch subcomponents.
       
        conv_layers = []
        prev_dim = self.input_shape[-1] 
        for out_dim,kern_size, stride, padding in self.conv_layers_spec: 

            conv_layers.append(nn.Conv2d(in_channels=prev_dim, out_channels=out_dim, kernel_size=kern_size, stride=stride, padding=padding))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_dim = out_dim

        conv_layers.append(nn.Conv2d(in_channels=self.conv_channel_sizes[-1], out_channels= 1, kernel_size=1, stride=1, padding=0))
        self.conv_layers = nn.Sequential(*conv_layers)

        conv_out_size = self._get_conv_output_size()

        self.fc_head = nn.Sequential(
            nn.Linear(self.additional_input_size, hidden_dim),
            nn.ReLU()
        )

        self.fc_final = nn.Sequential(
            nn.Linear(conv_out_size + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        



    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            x = self.conv_layers(dummy_input)
            return x.view(1, -1).size(1)

    def _forward(self, batch, **kwargs):
        
        data = batch[Columns.OBS]
        img = data['image']
        additional = data['additional']

        conv_res = self.conv_layers(img)
        conv_res = conv_res.view(conv_res.size(0), -1)
        fc_head_res = self.fc_head(additional)
        x = torch.cat([conv_res, fc_head_res], dim=1)
        action_logits = self.fc_final(x)

        # Return parameters for the default action distribution, which is
        # `TorchCategorical` (due to our action space being `gym.spaces.Discrete`).
        return {Columns.ACTION_DIST_INPUTS: action_logits}