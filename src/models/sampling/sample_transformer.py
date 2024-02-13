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

import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp


def _observe(state, encodings, model):
    variables = {'params': state.params, **state.model_state}
    cond, out = model.apply(variables, encodings,
                            method=model.encode)
    return cond, out['embeddings']


def _imagine(state, z_embeddings, actions, cond, t, rng, model, causal_masking):
    variables = {'params': state.params, **state.model_state}
    rng, new_rng = jax.random.split(rng)
    z, recon = model.apply(variables, z_embeddings, actions, cond, t, causal_masking,
                           method=model.sample_timestep,
                           rngs={'sample': rng})
    return recon, z, new_rng


def _decode(x, model):
    return model.vq_fns['decode'](x[:, None])[:, 0]


def _encode(x, model):
    return model.vq_fns['encode'](x)


def sample(model, state, video, actions, action_conditioned, open_loop_ctx,
           p_observe, p_imagine, p_encode, p_decode, seed=0, state_spec=None):

    use_xmap = state_spec is not None

    if use_xmap:
        num_local_data = max(1, jax.local_device_count() // model.config.num_shards)
    else:
        num_local_data = jax.local_device_count()
    rngs = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rngs, num_local_data)

    assert video.shape[0] == num_local_data, f'{video.shape}, {num_local_data}'
    assert model.config.n_cond <= open_loop_ctx

    if not model.config.use_actions:
        if actions is None:
            actions = jnp.zeros(video.shape[:3], dtype=jnp.int32)
        else:
            actions = jnp.zeros_like(actions)
    else:
        if not action_conditioned:
            actions = model.config.action_mask_id * np.ones(actions.shape,  dtype=jnp.int32)
    
    if video.shape[0] < jax.local_device_count():
        devices = jax.local_devices()[:video.shape[0]]
    else:
        devices = None
    # _, encodings = jax.pmap(model.vq_fns['encode'], devices=devices)(video)
    _, encodings = p_encode(video)
    
    # if use_xmap:
    #     p_observe = xmap(_observe, in_axes=(state_spec, ('data', ...)),
    #                     out_axes=('data', ...),
    #                     axis_resources={'data': 'dp', 'model': 'mp'})
    #     p_imagine = xmap(_imagine, in_axes=(state_spec, ('data', ...), ('data', ...),
    #                                         ('data', ...), (...,), ('data', ...)),
    #                     out_axes=('data', ...),
    #                     axis_resources={'data': 'dp', 'model': 'mp'})
    # else:
    #     p_observe = jax.pmap(_observe)
    #     p_imagine = jax.pmap(_imagine, in_axes=(0, 0, 0, 0, None, 0))
                     
    cond, zs = p_observe(state, encodings)

    if model.config.model in ["teco_transformer"]:
        sub = model.config.n_cond
    elif model.config.model in ["transformer", "performer"]:
        sub = 0
    else:
        raise NotImplementedError("The model type is not supported.")

    zs = zs[:, :, :model.config.seq_len - sub]

    recon = [encodings[:, :, i] for i in range(open_loop_ctx)]
    dummy_encoding = jnp.zeros_like(recon[0])
    itr = list(range(open_loop_ctx, model.config.eval_seq_len))
    for i in tqdm(itr):
        if i >= model.config.seq_len:
            encodings = jnp.stack([*recon[-model.config.seq_len + 1:], dummy_encoding], axis=2)
            cond, zs = p_observe(state, encodings)
            act = actions[:, :, i - model.config.seq_len + 1:i + 1]
            i = model.config.seq_len - 1
        else:
            act = actions[:, :, :model.config.seq_len]
        
        r, z, rngs = p_imagine(state, zs, act, cond, i, rngs)
        zs = zs.at[:, :, i - sub].set(z)
        recon.append(r)
    encodings = jnp.stack(recon, axis=2)

    def decode(samples):
        # samples: NBTHW
        N, B, T = samples.shape[:3]
        if N < jax.local_device_count():
            devices = jax.local_devices()[:N]
        else:
            devices = None

        samples = jax.device_get(samples)
        samples = np.reshape(samples, (-1, *samples.shape[3:]))

        recons = []
        for i in list(range(0, N * B * T, 64)):
            inp = samples[i:i + 64]
            inp = np.reshape(inp, (N, -1, *inp.shape[1:]))
            # recon = jax.pmap(_decode, devices=devices)(inp)
            recon = p_decode(inp)
            recon = jax.device_get(recon)
            recon = np.reshape(recon, (-1, *recon.shape[2:]))
            recons.append(recon)
        recons = np.concatenate(recons, axis=0)
        recons = np.reshape(recons, (N, B, T, *recons.shape[1:]))
        recons = np.clip(recons, -1, 1)
        return recons  # BTHWC
    samples = decode(encodings)

    if video.shape[3] == 16:
        video = decode(video)

    return samples, video 
