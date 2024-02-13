# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2022 BAIR OPEN RESEARCH COMMONS REPOSITORY
# To view a copy of this license, visit
# https://github.com/wilson1yan/teco/tree/master
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import lpips_jax

# Metrics below adapted from https://github.com/wilson1yan/teco/blob/bf56c2956515751bfd2b90355d52f7e362e288a1/teco/metrics.py


def compute_metric(prediction, ground_truth, metric_fn, average_dim=1):
    # BTHWC in [0, 1]
    assert prediction.shape == ground_truth.shape
    B, T = prediction.shape[0], prediction.shape[1]
    prediction = prediction.reshape(-1, *prediction.shape[2:])
    ground_truth = ground_truth.reshape(-1, *ground_truth.shape[2:])

    metrics = metric_fn(prediction, ground_truth)
    metrics = jnp.reshape(metrics, (B, T))
    metrics = metrics.mean(axis=average_dim)  # B or T depending on dim
    return metrics


# all methods below take as input pairs of images
# of shape BHWC. They DO NOT reduce batch dimension
# NOTE: Assumes that images are in [0, 1]
def get_ssim(pred, truth, average_dim=1):
    # output is shape bsz

    def fn(imgs1, imgs2):
        ssim_fn = ssim
        ssim_val = ssim_fn(imgs1, imgs2)
        return ssim_val

    return compute_metric(pred, truth, fn, average_dim=average_dim)


def ssim(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val, filter_size, filter_sigma, k1, k2)
    return jnp.mean(ssim_per_channel, axis=-1)


def _ssim_per_channel(img1, img2, max_val, filter_size, filter_sigma, k1, k2):
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = jnp.tile(kernel, [1, 1, img1.shape[-1], 1])
    kernel = jnp.transpose(kernel, [2, 3, 0, 1])

    compensation = 1.0

    def reducer(x):
        x_shape = x.shape
        x = jnp.reshape(x, (-1, *x.shape[-3:]))
        x = jnp.transpose(x, [0, 3, 1, 2])
        y = jax.lax.conv_general_dilated(x, kernel, [1, 1],
                                         'VALID', feature_group_count=x.shape[1])

        y = jnp.reshape(y, [*x_shape[:-3], *y.shape[1:]])
        return y

    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1, k2)
    ssim_val = jnp.mean(luminance * cs, axis=[-3, -2])
    cs = jnp.mean(cs, axis=[-3, -2])
    return ssim_val, cs


def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mean0 = reducer(x)
    mean1 = reducer(y)

    num0 = mean0 * mean1 * 2.0
    den0 = jnp.square(mean0) + jnp.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(x * y) * 2.0
    den1 = reducer(jnp.square(x) + jnp.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    return luminance, cs


def _fspecial_gauss(size, sigma):
    coords = jnp.arange(size, dtype=jnp.float32)
    coords -= (size - 1.0) / 2.0

    g = jnp.square(coords)
    g *= -0.5 / jnp.square(sigma)

    g = jnp.reshape(g, [1, -1]) + jnp.reshape(g, [-1, 1])
    g = jnp.reshape(g, [1, -1])
    g = jax.nn.softmax(g, axis=-1)
    return jnp.reshape(g, [size, size, 1, 1])


def get_psnr(pred, truth, average_dim=1):
    def fn(imgs1, imgs2):
        psnr_fn = psnr
        psnr_val = psnr_fn(imgs1, imgs2)
        return psnr_val
    return compute_metric(pred, truth, fn, average_dim=average_dim)


def psnr(a, b, max_val=1.0):
    mse = jnp.mean((a - b) ** 2, axis=[-3, -2, -1])
    val = 20 * jnp.log(max_val) / jnp.log(10.0) - jnp.float32(10 / jnp.log(10)) * jnp.log(mse)
    return val


def get_lpips(pred, truth, net='alexnet', average_dim=1):
    """net: ['alexnet', 'vgg16']"""
    lpips_eval = lpips_jax.LPIPSEvaluator(net=net, replicate=False)

    def fn(imgs1, imgs2):
        imgs1 = 2 * imgs1 - 1
        imgs2 = 2 * imgs2 - 1

        lpips = lpips_eval(imgs1, imgs2)
        lpips = jnp.reshape(lpips, (-1,))
        return lpips
    return compute_metric(pred, truth, fn, average_dim=average_dim)
