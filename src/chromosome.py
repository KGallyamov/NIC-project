from typing import List


class Chromosome:

    def __init__(self, cfg: List[str]):
        self.activation = cfg[0]
        self.layers_cfg = [[layer.split('_')[0], *map(int, layer.split('_')[1:])] for layer in cfg[1:]]

    def alter_layer(self, ind):

        return self

    def compress_layers(self, left, to_rm, right):

        return self

    def expand_layers(self, left, right):

        return self
# if __name__ == '__main__':
#     c = Chromosome(['ReLU', 'conv_128_64', 'conv_64_32'])
#     print(c.layers_cfg)
