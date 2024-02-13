# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------

from typing import Any, Optional, Tuple, Callable
import flax.linen as nn
from jax import lax, vmap
import jax.numpy as jnp


initializer = nn.initializers.variance_scaling(
    0.2, "fan_in", distribution="truncated_normal"
)


class Half_GLU(nn.Module):
    dim: int = 256

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = nn.LayerNorm(name="norm")(inputs)
        x = nn.gelu(x)
        x2 = nn.Dense(self.dim, kernel_init=initializer)(x)
        x = x * nn.sigmoid(x2)
        return x


class SEBlock(nn.Module):
    """Applies Squeeze-and-Excitation."""
    act: Callable = nn.relu
    axis: Tuple[int, int] = (-3, -2)
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        hidden_size = max(x.shape[-1] // 16, 4)
        y = x.mean(axis=self.axis, keepdims=True)
        y = nn.Dense(features=hidden_size, dtype=self.dtype, name='reduce')(y)
        y = self.act(y)
        y = nn.Dense(features=x.shape[-1], dtype=self.dtype, name='expand')(y)
        return nn.sigmoid(y) * x


class ResnetBlock(nn.Module):
    activation: nn.module  # swish or gelu
    k_size: int
    use_conv_shortcut: bool = False
    out_channels: Optional[int] = None
    num_groups: int = 32
    squeeze_excite: bool = False

    @nn.compact
    def __call__(self, x):
        # x is shape (BHWC)
        out_channels = self.out_channels or x.shape[-1]

        h = x
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = self.activation(h)
        h = nn.Conv(out_channels, [self.k_size, self.k_size],
                    padding='SAME')(h)

        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = self.activation(h)
        h = nn.Conv(out_channels, [self.k_size, self.k_size],
                    padding='SAME')(h)
        if self.squeeze_excite:
            h = SEBlock()(h)

        if x.shape[-1] != out_channels:
            if self.use_conv_shortcut:
                x = nn.Conv(out_channels, [self.k_size, self.k_size],
                            padding='SAME')(x)
            else:
                x = nn.Conv(out_channels, [1, 1])(x)
        return self.activation(x + h)


# For vmapping ResnetBlock across sequence length
VmapResnetBlock = nn.vmap(ResnetBlock,
                          in_axes=(0),
                          out_axes=(0),
                          variable_axes={"params": None,
                                         "dropout": None},
                          split_rngs={"params": False,
                                      "dropout": False})


class DiagResnetBlock(nn.Module):
    activation: nn.module  # swish or gelu
    k_size: int
    use_conv_shortcut: bool = False
    out_channels: Optional[int] = None
    num_groups: int = 32
    squeeze_excite: bool = False

    @nn.compact
    def __call__(self, x):
        # x is shape (BHWC)
        out_channels = self.out_channels or x.shape[-1]

        h = x
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = self.activation(h)
        h = nn.Conv(out_channels, [self.k_size, self.k_size],
                    padding='SAME')(h)

        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = self.activation(h)
        if self.squeeze_excite:
            h = SEBlock()(h)

        if x.shape[-1] != out_channels:
            if self.use_conv_shortcut:
                x = nn.Conv(out_channels, [self.k_size, self.k_size],
                            padding='SAME')(x)
            else:
                x = nn.Conv(out_channels, [1, 1])(x)
        return self.activation(x + h)


# For vmapping ResnetBlock across sequence length
VmapDiagResnetBlock = nn.vmap(DiagResnetBlock,
                              in_axes=(0),
                              out_axes=(0),
                              variable_axes={"params": None,
                                             "dropout": None},
                              split_rngs={"params": False,
                                          "dropout": False})


class BasicConv(nn.Module):
    k_size: int
    out_channels: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        # x is shape (BHWC)
        out_channels = self.out_channels or x.shape[-1]
        return nn.Conv(out_channels, [self.k_size, self.k_size],
                       padding='SAME')(x)


# For vmapping FlaxConv across sequence length
VmapBasicConv = nn.vmap(BasicConv,
                        in_axes=(0),
                        out_axes=(0),
                        variable_axes={"params": None},
                        split_rngs={"params": False})


class ConvNormNL(nn.Module):
    k_size: int
    activation: nn.module
    out_channels: Optional[int] = None
    num_groups: int = 32
    squeeze_excite: bool = False

    @nn.compact
    def __call__(self, x):
        # x is shape (BHWC)
        out_channels = self.out_channels or x.shape[-1]
        h = nn.GroupNorm(num_groups=self.num_groups)(x)
        h = self.activation(h)
        h = nn.Conv(out_channels, [self.k_size, self.k_size],
                    padding='SAME')(h)
        if self.squeeze_excite:
            h = SEBlock()(h)
        return h


# For vmapping FlaxConv across sequence length
VmapConvNormNL = nn.vmap(ConvNormNL,
                         in_axes=(0),
                         out_axes=(0),
                         variable_axes={"params": None},
                         split_rngs={"params": False})


class Basic_CD_Conv(nn.Module):
    activation: nn.module
    k_C: int
    k_D: int
    out_channels: Optional[int] = None
    num_groups: int = 32
    squeeze_excite: bool = False

    @nn.compact
    def __call__(self, x, u):
        # x is shape (BHWC)
        out_channels = self.out_channels or x.shape[-1]

        h = x
        h = nn.Conv(out_channels, [self.k_C, self.k_C], padding='SAME')(h)
        h += nn.Conv(out_channels, [self.k_D, self.k_D], padding='SAME')(u)
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = self.activation(h)
        if self.squeeze_excite:
            h = SEBlock()(h)
        return h


# For vmapping FlaxConv across sequence length
VmapBasic_CD_Conv = nn.vmap(Basic_CD_Conv,
                            in_axes=(0, 0),
                            out_axes=(0),
                            variable_axes={"params": None},
                            split_rngs={"params": False})


class Diag_CD_Conv(nn.Module):
    activation: nn.module
    k_D: int
    out_channels: Optional[int] = None
    num_groups: int = 32
    squeeze_excite: bool = False

    @nn.compact
    def __call__(self, x, u):
        # x is shape (BHWC)
        out_channels = self.out_channels or x.shape[-1]

        h = x
        h += nn.Conv(out_channels, [self.k_D, self.k_D], padding='SAME')(u)
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = self.activation(h)
        if self.squeeze_excite:
            h = SEBlock()(h)
        return h


# For vmapping FlaxConv across sequence length
VmapDiag_CD_Conv = nn.vmap(Diag_CD_Conv,
                           in_axes=(0, 0),
                           out_axes=(0),
                           variable_axes={"params": None},
                           split_rngs={"params": False})


def vmap_conv(B, us):
    """Performs a convolution at each timestep of a sequence using vmap
       to vectorize across the sequence length.
       Args:
            B (float32):   conv kernel            (k_B, k_B, U, P)
            us (float 32): input sequence         (L, bsz, h_u, w_u, U)
       Returns:
            Sequence of convolved inputs Bu (float32)  (L, bsz, h_u, w_u, P)
            )
    """
    def input_to_state_conv(B, u):
        # Performs the input to state convolution for a single timestep
        return lax.conv_general_dilated(u, B, (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

    return vmap(input_to_state_conv, in_axes=(None, 0))(B, us)


def merge_1x1_kernels_jax(k1, k2):
    """
    The merge function using Jax conv operators is specialized to 1x1 kernels
    and removes an unecessary flip

    :input k1: A tensor of shape ``(1,1, in1, out1)``
    :input k2: A tensor of shape ``(1,1, in2, out2)``
    :returns: A tensor of shape  ``(1,1, in3, out3)``
      so that convolving an image with it equals convolving with k1 and
      then with k2.

    Note that we transpose k1 to adapt to NHWC format, i.e. we will
    treat the input dim of k1 as the batch dim and the output dim as the input
    dim for the k2 kernel
    """
    # k1 is HWIO
    k3 = lax.conv_general_dilated(k1.transpose(2, 0, 1, 3),  # lhs = NHWC image tensor
                                  k2,  # rhs = HWIO conv kernel tensor
                                  (1, 1),  # window strides
                                  'VALID',
                                  dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    # k3 is NHWC
    return k3.transpose(1, 2, 0, 3)  # permute to adapt to HWIO
