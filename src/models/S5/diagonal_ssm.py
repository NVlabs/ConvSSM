# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2022 Linderman Lab
# To view a copy of this license, visit
# https://github.com/lindermanlab/S5/blob/main/LICENSE
# ------------------------------------------------------------------------------


from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh
from jax import random
from jax.scipy.linalg import block_diag

from . import diagonal_scans


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_Normal_HiPPO(N):
    """normal approximation to the HiPPO-LegS matrix"""
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)
    nhippo = hippo + P[:, np.newaxis] * P[np.newaxis, :]

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)

    return nhippo, P, B


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    U, dt_min, dt_max = input
    log_steps = []
    for i in range(U):
        key, skey = jax.random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return np.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


class S5SSM(nn.Module):
    Lambda_re_init: np.DeviceArray
    Lambda_im_init: np.DeviceArray
    V: np.DeviceArray
    Vinv: np.DeviceArray
    clip_eigs: bool
    parallel: bool  # Compute scan in parallel

    U: int    # Number of SSM input and output features
    P: int    # Number of state features of SSM

    dt_min: float  # for initializing discretization step
    dt_max: float

    def setup(self):

        local_P = 2 * self.P

        # Initialize diagonal state to state transition kernel Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) and output to state (C) matrices
        B_init = lecun_normal()
        B_shape = (local_P, self.U)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        C_init = trunc_standard_normal
        C_shape = (self.U, local_P, 2)

        self.C = self.param("C",
                            lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                            C_shape)

        self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.U,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = np.exp(self.log_step[:, 0])

        # Nonlinear activation
        self.out2 = nn.Dense(self.U)

        if self.parallel:
            # Discretize
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        else:
            # trick to cache the discretization for step-by-step
            # generation
            def init_discrete():
                Lambda_bar, B_bar = discretize_zoh(self.Lambda, B_tilde, step)
                return Lambda_bar, B_bar
            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

    def __call__(self, input_sequence, x0):
        """
        input sequence is shape (L, bsz, H, W, U)
        x0 is (bsz, H, W, U)
        Returns:
            x_L (float32): the last state of the SSM  (bsz, H, W, P)
            ys (float32): the conv SSM outputs       (L,bsz, H, W, U)
        """
        if self.parallel:
            x_last, ys = diagonal_scans.apply_ssm_parallel(self.Lambda_bar,
                                                           self.B_bar,
                                                           self.C_tilde,
                                                           input_sequence,
                                                           x0)

        else:
            # For sequential generation (e.g. autoregressive decoding)
            x_last, ys = diagonal_scans.apply_ssm_sequential(*self.ssm,
                                                             self.C_tilde,
                                                             input_sequence,
                                                             x0)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        ys = ys + Du

        ys = nn.gelu(ys)
        ys = ys * jax.nn.sigmoid(self.out2(ys))

        return x_last, ys


def hippo_initializer(ssm_size, blocks):
    block_size = int(ssm_size/blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
    ssm_size = ssm_size // 2
    block_size = block_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))

    return Lambda.real, Lambda.imag, V, Vinv, ssm_size


def init_S5SSM(ssm_size,
               blocks,
               clip_eigs,
               U,
               dt_min,
               dt_max):
    Lambda_re_init, Lambda_im_init,\
        V, Vinv, ssm_size = hippo_initializer(ssm_size, blocks)

    return partial(S5SSM,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   clip_eigs=clip_eigs,
                   U=U,
                   P=ssm_size,
                   dt_min=dt_min,
                   dt_max=dt_max)
