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


def _observe(state, video, actions, model_par):
    variables = {'params': state.params, **state.model_state}
    outs = model_par.apply(variables,
                           video,  actions,
                           method=model_par.condition)

    _, z_embeddings, _, _, last_states = outs
    z_T = z_embeddings[:, -1:]
    return z_T, last_states


def _imagine(state, z_embedding, initial_states, action, rng, model_seq, causal_masking):
    variables = {'params': state.params, **state.model_state}
    rng, new_rng = jax.random.split(rng)
    out, _ = model_seq.apply(variables,
                             z_embedding, initial_states, action, causal_masking,
                             method=model_seq.sample_timestep,
                             rngs={'sample': rng},
                             mutable=["prime"])

    z_t, _, recon, last_states = out
    return recon, z_t, last_states, new_rng


def _decode(x, model_par):
    return model_par.vq_fns['decode'](x[:, None])[:, 0]


def _encode(x, model_par):
    return model_par.vq_fns['encode'](x)


def sample(model_par, state, video, actions, action_conditioned, open_loop_ctx,
           p_observe, p_imagine, p_encode, p_decode,
           seed=0, state_spec=None):

    use_xmap = state_spec is not None

    if use_xmap:
        num_local_data = max(1, jax.local_device_count() // model_par.config.num_shards)
    else:
        num_local_data = jax.local_device_count()
    rngs = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rngs, num_local_data)

    assert video.shape[0] == num_local_data, f'{video.shape}, {num_local_data}'
    assert model_par.config.n_cond <= model_par.config.open_loop_ctx

    if not model_par.config.use_actions:
        if actions is None:
            actions = jnp.zeros(video.shape[:3], dtype=jnp.int32)
        else:
            actions = jnp.zeros_like(actions)
    else:
        if not action_conditioned:
            actions = model_par.config.action_mask_id * np.ones(actions.shape,  dtype=jnp.int32)
    
    if video.shape[0] < jax.local_device_count():
        devices = jax.local_devices()[:video.shape[0]]
    else:
        devices = None

    num_input_frames = open_loop_ctx
    _, encodings = p_encode(video[:, :, :num_input_frames])

    z, last_states = p_observe(state, encodings, actions[:, :, :num_input_frames])

    recon = [encodings[:, :, i] for i in range(num_input_frames)]
    dummy_encoding = jnp.zeros_like(recon[0])
    itr = list(range(num_input_frames, model_par.config.eval_seq_len))
    for i in tqdm(itr):
        if i >= model_par.config.seq_len:
            # TODO
            pass
        else:
            act = actions[:, :, i:i+1]
        
        r, z, last_states, rngs = p_imagine(state, z, last_states, act, rngs)
        z = jnp.expand_dims(z, 2)
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
