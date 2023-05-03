# Default libraries
from typing import List, Tuple
from collections.abc import Iterable

import torch
# Requires installation (check requirements.txt)
import torch.nn as nn

# Our units
from src.constants import ACTIVATIONS


def iterate(iterable):
    """
    Generator that iterate through any-d array

    :param iterable:  any iterable object containing non-iterable objects as leafs (e.x. [[1], 2, [[3]]]
    :return:          each non-iterable elements 1 by 1 (e.x. [1, 2, 3])
    """

    for elem in iterable:
        if isinstance(elem, Iterable):
            yield from iterate(elem)
        else:
            yield elem


def _resolve_act(activation: str) -> nn.Module:
    """
    Given activation name, resolve it into the corresponding object

    :param activation:  str, activation name
    :return:            activation module
    """
    assert activation in ACTIVATIONS, f'Activation {activation} is not supported'
    act = None
    if activation.lower() == 'relu':
        act = nn.ReLU()
    elif activation.lower() == 'tanh':
        act = nn.Tanh()
    elif activation.lower() == 'sigmoid':
        act = nn.Sigmoid()
    else:  # activation.lower() == 'lrelu':
        act = nn.LeakyReLU()
    return act


def _resolve_layer(layer_cfg: str, activation: str) -> Tuple[List[nn.Module], List[nn.Module]]:
    """
    Given layer config, return the corresponding AE blocks for encoder and decoder

    :param layer_cfg:   str in the format layer_fanin_fanout (_kernelsize)
    :param activation:  activation name encoding
    :return:            encoder & decoder
    """
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
    def __init__(self, cfg: List[str], image_shape: tuple[int, int]):
        """
        :param cfg: List of str in the format: layertype_fanin_fanout (_kernelsize for layertype=conv)
        """
        super().__init__()
        self.cfg = cfg
        activation = cfg[0]
        encoder_list = []
        decoder_list = []

        for i in range(1, len(cfg)):
            layer_cfg = cfg[i]
            enc_layer = []
            dec_layer = []
            enc_, dec_ = _resolve_layer(layer_cfg, activation)

            # Add symmetric layers to encoder and decoder
            enc_layer.append(enc_)
            dec_layer.append(dec_)
            if 'conv' in cfg[i - 1].lower() and 'linear' in cfg[i].lower():
                assert NotImplementedError
                # enc_layer.insert(0, [nn.Flatten()])
                # dec_layer.append([nn.Unflatten(1, (int(cfg[i - 1].split('_')[2]), ??, ??))])

            # save layers
            encoder_list.append(enc_layer)
            decoder_list.append(dec_layer)  # will be reversed further (just improve performance)

        decoder_list.reverse()  # should be in increasing order, not decreasing

        # define encoder/decoder
        self.encoder = nn.Sequential(*list(iterate(encoder_list)))
        self.decoder = nn.Sequential(*list(iterate(decoder_list)))

    def forward(self, x: torch.Tensor):
        x_shape = x.shape
        if 'linear' in self.cfg[1]:
            x = torch.reshape(x, (x.shape[0], -1))
        t = self.encoder(x)
        _x = self.decoder(t)

        return torch.reshape(_x, x_shape), t

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
