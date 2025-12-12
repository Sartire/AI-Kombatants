import torch
import torch.nn as nn


import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule

from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.apis import ValueFunctionAPI

from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.utils.typing import TensorType

# Define a custom env class by subclassing `TorchRLModule`.
# Also subclass `ValueFunctionAPI` to implement a value function for PPO
class Kombatant(TorchRLModule, ValueFunctionAPI):
    def __init__(self, *args, **kwargs):
        # documentation says not to override __init__, but it seems like we need to
        TorchRLModule.__init__(self, *args, **kwargs)

        self.setup()

    @override(TorchRLModule)    
    def setup(self):
        # You have access here to the following already set attributes:
        # self.observation_space
        # self.action_space
        # self.inference_only
        # self.model_config  # <- a dict with custom settings
        
        self.input_shape = self.observation_space['image'].shape
        self.additional_input_size = self.observation_space['additional_data'].shape[-1]
        
        ### model parameters from config
        # size after merging conv and the extra info
        hidden_dim = self.model_config["hidden_dim"]
        # lstm output size
        embedding_dim = self.model_config["embedding_dim"]
        # action space
        output_dim = self.action_space.n
        # list of tuples (out_dim, kernel_size, stride, padding)
        self.conv_layers_spec = self.model_config['conv_layers_spec']
        # True/False
        self.conv_normalize = self.model_config['conv_normalize']

        # lstm params -- not sure we can do more than 1 layer without passing sequences
        num_lstm_layers =  1 #self.model_config['num_lstm_layers']
        
        # Define and assign torch subcomponents.
        # remember batch is dim 0

        # build conv layers from config spec
        conv_layers = []
        prev_dim = self.input_shape[-1] 
        for out_dim,kern_size, stride, padding in self.conv_layers_spec: 

            conv_layers.append(nn.Conv2d(in_channels=prev_dim, out_channels=out_dim, kernel_size=kern_size, stride=stride, padding=padding))
            if self.conv_normalize:
                conv_layers.append(nn.GroupNorm(8, out_dim))
            conv_layers.append(nn.ReLU())
            
            prev_dim = out_dim

        # merge into a convolutional block
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # combine conv and extra info, go through fc layers
        conv_out_size = self._get_conv_output_size()

        self.fc_head = nn.Sequential(
            nn.Linear(self.additional_input_size, hidden_dim // 2),
            nn.ReLU()
        )

        self.fc_final = nn.Sequential(
            nn.Linear(conv_out_size + hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )

        # add lstm
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=embedding_dim, batch_first=True,
                            num_layers = num_lstm_layers )
        


        # policy and value heads
        self.policy_head = nn.Linear(embedding_dim, output_dim)
        
        self.value_head = nn.Linear(embedding_dim, 1)
        


    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_input = dummy_input.permute(0, 3, 1, 2)
            x = self.conv_layers(dummy_input)
            return x.view(1, -1).size(1)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        
        embeddings, state_out = self._compute_embeddings_and_state_outs(batch)
        action_logits = self.policy_head(embeddings)
        
        # Return parameters for the default action distribution, which is
        # `TorchCategoricalDistribution` (action space is `gym.spaces.Discrete`).
        return {Columns.ACTION_DIST_INPUTS: action_logits,
                Columns.STATE_OUT: state_out}
    
    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings, state_out = self._compute_embeddings_and_state_outs(batch)
        
        action_logits = self.policy_head(embeddings)

        # Return features and logits as ACTION_DIST_INPUTS (categorical distribution).
        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.EMBEDDINGS: embeddings,
            Columns.STATE_OUT: state_out
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch,
        embeddings = None,
    ) -> TensorType:
        # Features not provided -> We need to compute them first.
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_state_outs(batch)

        return self.value_head(embeddings).squeeze(-1)
    
    def _compute_embeddings_and_state_outs(self, batch):

        data = batch[Columns.OBS]
        img = data['image']
        additional = data['additional_data']

        state_in = batch[Columns.STATE_IN]

        # run convolution -----
        # move channels to dim 1
        if img.shape[1] != self.input_shape[-1]:
            img = img.permute(0, 3, 1, 2)
        

        conv_res = self.conv_layers(img)
        
        # flatten conv output
        conv_res = torch.flatten(conv_res, start_dim=1) 
        fc_head_res = self.fc_head(additional)

        merged = torch.cat([conv_res, fc_head_res], dim=1)

        merged = self.fc_final(merged)

        # pass to lstm
        h0, c0 = state_in['h'], state_in['c']

        embeddings, (h1,c1) = self.lstm(merged, (h0.unsqueeze(0), c0.unsqueeze(0)))

        state_out = {
            'h': h1.squeeze(0),
            'c': c1.squeeze(0)
        }

        return embeddings, state_out
