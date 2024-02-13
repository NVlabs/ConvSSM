# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------


from functools import partial
from flax import linen as nn
from jax.nn.initializers import he_normal

from . import scans


def initialize_kernel(key, shape):
    """For general kernels, e.g. C,D, encoding/decoding"""
    out_dim, in_dim, k = shape
    fan_in = in_dim*(k**2)

    # Note in_axes should be the first by default:
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling
    return he_normal()(key,
                       (fan_in, out_dim)).reshape(k,
                                                  k,
                                                  in_dim,
                                                  out_dim)


class ConvLSTM(nn.Module):
    U: int    # Number of SSM input and output features
    P: int    # Number of state features of SSM
    k_A: int  # A kernel width/height
    parallel: bool = False  # Cannot compute convLSTM in parallel
                            # but include this attribute for consistency
                            # in layers.py

    def setup(self):
        # Initialize state to state (A) transition kernel
        self.A = self.param("A",
                            initialize_kernel,
                            (4 * self.P, self.U+self.P,  self.k_A))

    def __call__(self, input_sequence, x0):
        """
        input sequence is shape (L, bsz, U, H, W)
        x0 is (bsz, U, H, W)
        Returns:
            x_L (float32): the last state of the SSM  (bsz, P, H, W)
            hs (float32): the conv LSTM states       (L,bsz, U, H, W)
        """
        # For sequential generation (e.g. autoregressive decoding)
        return scans.apply_convLSTM(self.A,
                                    input_sequence,
                                    x0)


def init_ConvLSTM(U,
                  P,
                  k_A):
    return partial(ConvLSTM,
                   U=U,
                   P=P,
                   k_A=k_A)
