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


import jax.numpy as jnp

from src.models.sequence_models.VQ.S5 import S5
from src.models.sequence_models.noVQ.convLSTM import CONVLSTM_NOVQ
from src.models.sequence_models.noVQ.convS5 import CONVS5_NOVQ
from src.models.sequence_models.VQ.teco_S5 import TECO_S5
from src.models.sequence_models.VQ.teco_convS5 import TECO_CONVS5
from src.models.sequence_models.VQ.convS5 import CONVS5
from src.models.sequence_models.VQ.teco_transformer import TECO_TRANSFORMER
from src.models.sequence_models.VQ.transformer import TRANSFORMER
from src.models.sequence_models.noVQ.transformer import TRANSFORMER_NOVQ
from .vqgan import VQGAN
from .vae import VAE
from functools import partial


def load_vqvae(ckpt_path, need_encode=True):
    import jax
    import argparse

    model, state = load_ckpt(ckpt_path, training=False, replicate=False)

    def wrap_apply(fn):
        variables = {'params': state.params, **state.model_state}
        return lambda *args: model.apply(variables, *args, method=fn)

    def no_encode(encodings):
        variables = {'params': state.params, **state.model_state}
        embeddings = model.apply(variables, encodings, method=model.codebook_lookup) 
        return embeddings, encodings

    video_encode = jax.jit(wrap_apply(model.encode)) if need_encode else jax.jit(no_encode)
    video_decode = jax.jit(wrap_apply(model.decode))
    codebook_lookup = jax.jit(wrap_apply(model.codebook_lookup))

    return dict(encode=video_encode, decode=video_decode, lookup=codebook_lookup), argparse.Namespace(latent_shape=model.latent_shape, embedding_dim=model.embedding_dim, n_codes=model.n_codes)


def load_ckpt(ckpt_path, replicate=True, return_config=False, 
              default_if_none=dict(), need_encode=None, **kwargs):
    import os.path as osp
    import pickle
    from flax import jax_utils
    from flax.training import checkpoints
    from ..train_utils import TrainState

    config = pickle.load(open(osp.join(ckpt_path, 'args'), 'rb'))
    for k, v in kwargs.items():
        setattr(config, k, v)
    for k, v in default_if_none.items():
        if not hasattr(config, k):
            print('did not find', k, 'setting default to', v)
            setattr(config, k, v)
    
    model = get_model(config, need_encode=need_encode)
    state = checkpoints.restore_checkpoint(osp.join(ckpt_path, 'checkpoints'), None)
    if config.model in ['teco_convS5',  'convS5', 'teco_S5', 'S5', 'convS5_noVQ', 'convLSTM_noVQ']:
        state = TrainState(
            step=state['step'],
            params=state['params'],
            opt_state=state['opt_state'],
            model_state=state['model_state'],
            apply_fn=model(parallel=True, training=True).apply,
            tx=None
        )
    else:
        state = TrainState(
            step=state['step'],
            params=state['params'],
            opt_state=state['opt_state'],
            model_state=state['model_state'],
            apply_fn=model.apply,
            tx=None
        )

    assert state is not None, f'No checkpoint found in {ckpt_path}'

    if replicate:
        state = jax_utils.replicate(state)

    if return_config:
        return model, state, config
    else:
        return model, state


def get_model(config, need_encode=None, xmap=False, **kwargs):
    if config.model in ['teco_transformer',  'transformer', 
                        'teco_convS5', 'convS5',
                        'S5', 'teco_S5']:
        if need_encode is None:
            need_encode = not 'encoded' in config.data_path
        vq_fns, vqvae = load_vqvae(config.vqvae_ckpt, need_encode)
        kwargs.update(vq_fns=vq_fns, vqvae=vqvae)

    kwargs['dtype'] = jnp.float32

    if config.model == 'vqgan':
        model = VQGAN(config, **kwargs)
    elif config.model == 'autoencoder':
        model = VAE(config, **kwargs)
    elif config.model == 'transformer':
        model = TRANSFORMER(config, **kwargs)
    elif config.model == 'teco_transformer':
        model = TECO_TRANSFORMER(config, **kwargs)
    elif config.model == 'transformer_noVQ':
        model = TRANSFORMER_NOVQ(config, **kwargs)
    elif config.model == 'convS5':
        model = partial(CONVS5,
                        config=config, **kwargs)
    elif config.model == 'convS5_noVQ':
        model = partial(CONVS5_NOVQ,
                        config=config, **kwargs)
    elif config.model == 'teco_convS5':
        model = partial(TECO_CONVS5,
                        config=config, **kwargs)
    elif config.model == 'S5':
        model = partial(S5,
                        config=config, **kwargs)
    elif config.model == 'teco_S5':
        model = partial(TECO_S5,
                        config=config, **kwargs)
    elif config.model == 'convLSTM_noVQ':
        model = partial(CONVLSTM_NOVQ,
                        config=config, **kwargs)
    else:
        raise ValueError(f'Invalid model: {config.model}')

    return model
