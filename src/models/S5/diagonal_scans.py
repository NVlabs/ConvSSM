# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2022 Linderman Lab
# To view a copy of this license, visit
# https://github.com/lindermanlab/S5/blob/main/LICENSE
# ------------------------------------------------------------------------------

import jax
from jax import lax, numpy as np


@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm_parallel(Lambda_bar, B_bar, C_tilde, input_sequence, x0):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            x0             (complex64): initial state                    (P,)
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)
    Bu_elements = Bu_elements.at[0].add(Lambda_bar * x0)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    ys = jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    return xs[-1], ys


def apply_ssm_sequential(Lambda_bar, B_bar, C_tilde, input_sequence, x0):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            x0         (complex64):  initial state                       (P,)
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    def step(x_k_1, u_k):
        Bu = B_bar @ u_k
        x_k = Lambda_bar * x_k_1 + Bu
        y_k = 2*(C_tilde @ x_k).real
        return x_k, y_k

    return lax.scan(step, x0, input_sequence)
