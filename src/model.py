import numpy as np
import torch
import torch.nn as nn

from typing import List
from collections import OrderedDict


def _resolve_act(activation: str) -> nn.Module:
    act = None
    if activation.lower() == 'relu':
        act = nn.ReLU()
    elif activation.lower() == 'tanh':
        act = nn.Tanh()
    elif activation.lower() == 'sigmoid':
        act = nn.Sigmoid()
    elif activation.lower() == 'lrelu':
        act = nn.LeakyReLU
    else:
        assert NotImplementedError, f'Activation {activation} is not supported'
    return act


def _resolve_layer(layer_cfg: str, activation):
    l_type = layer_cfg.split('_')[0]
    enc_layer, dec_layer = None, None
    fan_in, fan_out = map(int, layer_cfg.split('_')[1:3])
    if l_type.lower() == 'conv':
        kernel_size = int(layer_cfg.split('_')[-1])
        enc_layer = [
            nn.Conv2d(fan_in, fan_out, kernel_size),
            _resolve_act(activation),
            nn.BatchNorm2d(fan_out),
        ]
        dec_layer = [
            nn.ConvTranspose2d(fan_out, fan_in, kernel_size),
            _resolve_act(activation),
            nn.BatchNorm2d(fan_in),
        ]

    elif l_type.lower() == 'linear':
        enc_layer = [nn.Linear(fan_in, fan_out), _resolve_act(activation)]
        dec_layer = [nn.Linear(fan_out, fan_in), _resolve_act(activation)]
    else:
        assert NotImplementedError, f'Module {l_type} is not supported'

    return enc_layer, dec_layer


class AutoEncoder(nn.Module):
    def __init__(self, cfg: List[str]):
        super().__init__()
        activation = cfg[0]
        encoder_list = []
        decoder_list = []

        for i in range(1, len(cfg)):
            layer_cfg = cfg[i]
            enc_layer = []
            dec_layer = []
            enc_, dec_ = _resolve_layer(layer_cfg, activation)
            enc_layer.append(enc_)
            dec_layer.append(dec_)
            if 'conv' in cfg[i - 1].lower() and 'linear' in cfg[i]:
                assert NotImplementedError
                # enc_layer.insert(0, [nn.Flatten()])
                # dec_layer.append([nn.Unflatten(1, (int(cfg[i - 1].split('_')[2]), ??, ??))])

            encoder_list = encoder_list + enc_layer
            decoder_list = dec_layer + decoder_list

        self.encoder = nn.Sequential(*[l for block in encoder_list for l in block])
        self.decoder = nn.Sequential(*[l for block in decoder_list for l in block])

    def forward(self, x):
        t = self.encoder(x)
        _x = self.decoder(t)
        return _x


# if __name__ == '__main__':
#     cfg_sample = ['ReLU', 'conv_3_32_3', 'conv_32_64_3']
#     ae = AutoEncoder(cfg_sample)
#     rnd = np.random.random((16, 3, 64, 64))
#     print(rnd.shape)
#     # print(ae.encoder)
#     # print(ae.decoder)
#     print(ae(torch.from_numpy(rnd).float()).shape)
#     cfg_sample = ['ReLU', 'linear_128_64', 'linear_64_32']
#     ae = AutoEncoder(cfg_sample)
#     rnd = np.random.random((16, 128))
#     print(rnd.shape)
#     # print(ae.encoder)
#     # print(ae.decoder)
#     print(ae(torch.from_numpy(rnd).float()).shape)
