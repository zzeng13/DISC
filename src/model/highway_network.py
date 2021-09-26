import torch.nn as nn
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
    def __init__(self, config):
        super(HighwayNetwork, self).__init__()
        self.config = config
        self.n_layers = config.HIGHWAY_NUM_LAYERS
        in_size = config.CHAR_EMBED_CNN_NUM_OUT_CHANNELS + config.PRETRAINED_GLOVE_EMBED_DIM
        self.normal_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(self.n_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(self.n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            normal_layer_ret = F.relu(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x

        return x