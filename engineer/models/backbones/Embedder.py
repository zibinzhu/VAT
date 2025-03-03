import torch
import torch.nn as nn
from engineer.models.registry import BACKBONES
from .base_backbone import _BaseBackbone
""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

@BACKBONES.register_module
class Embedder(_BaseBackbone):
    def __init__(self, include_input, input_dims, max_freq_log2, num_freqs, log_sampling, periodic_fns):
        super(Embedder, self).__init__()
        self.name = 'Embedder Backbone'
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.embed_fns, self.out_dim = self.create_embedding_fn()

        self.input_para={'include_input':include_input,'input_dims':input_dims,'max_freq_log2':max_freq_log2,'num_freqs':num_freqs,
                         'log_sampling':log_sampling,'periodic_fns': periodic_fns}

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        return embed_fns, out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
