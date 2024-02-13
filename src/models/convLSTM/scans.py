# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------

from jax import lax, numpy as np
from jax.nn import sigmoid


# Scan functions
def apply_convLSTM(A, us, x0):
    """Compute the output sequence of the convolutional LSTM
        given the input sequence sequentially. For testing purposes.
    Args:
        A (float32): Conv kernel A                (k_a,k_a, U+P, 4*P)
        us (float32): input sequence of features  (L,bsz,H, W, U)
        x0 (float32): initial state               (bsz, H, W, P)
    Returns:
        x_L (float32): the last state of the SSM  (bsz, H, W, P)
        ys (float32): the conv LSTM states     (L,bsz, H, W, U)
    """

    def step(x_k_1, u_k):
        c_k_1, h_k_1 = x_k_1

        combo = np.concatenate((u_k, h_k_1), axis=-1)  # concat along channel dim

        combo_conv = lax.conv_general_dilated(combo, A, (1, 1),
                                              'SAME',
                                              dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        # combo_conv = lax.conv(combo, A, (1, 1), 'SAME')
        cc_i, cc_f, cc_o, cc_g = np.split(combo_conv, 4, axis=-1)

        i = sigmoid(cc_i)
        f = sigmoid(cc_f)
        o = sigmoid(cc_o)
        g = np.tanh(cc_g)

        c_k = f * c_k_1 + i * g
        h_k = o * np.tanh(c_k)
        return (c_k, h_k), h_k
    return lax.scan(step, x0, us)
