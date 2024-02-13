# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------


import os
import os.path as osp
import numpy as np
import argparse
import yaml
import pickle
import glob
from functools import partial

import jax
from jax import random, lax
from flax.training import checkpoints
from flax import jax_utils

from src.data import Data
from src.train_utils import init_model_state, \
        get_first_device, seed_all
from src.models import get_model
from src.models.sampling import sample_convSSM, sample_transformer, sample_convSSM_noVQ, sample_transformer_noVQ


def save(i, s, r, folder, open_loop_ctx):
    s = s.reshape(-1, *s.shape[2:])
    s = s * 0.5 + 0.5
    s = (s * 255).astype(np.uint8)
    r = r.reshape(-1, *r.shape[2:])
    r = r * 0.5 + 0.5
    r = (r * 255).astype(np.uint8)

    s[:, :open_loop_ctx] = r[:, :open_loop_ctx]

    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(osp.join(folder, f'data_{i}.npz'), real=r, fake=s)


def main():
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    seed_all(config.seed)

    files = glob.glob(osp.join(config.output_dir, 'checkpoints', '*'))
    if len(files) > 0:
        print('Found previous checkpoints', files)
        config.ckpt = config.output_dir
    else:
        config.ckpt = None

    data = Data(config)
    train_loader = data.create_iterator(train=True)
    test_loader = data.create_iterator(train=False)

    steps_per_eval = int(config.eval_size/config.batch_size)

    batch = next(test_loader)
    batch = get_first_device(batch)
    model = get_model(config)

    if config.model in ["teco_convS5", "convS5", "teco_S5", "S5"]:
        model_par_eval = model(parallel=True, training=False)
        model_seq_eval = model(parallel=False, training=False)
        model = model(parallel=True, training=True)

        p_observe = jax.pmap(partial(sample_convSSM._observe,
                                     model_par=model_par_eval))
        if config.causal_masking:
            p_imagine_1 = jax.pmap(partial(sample_convSSM._imagine,
                                           model_seq=model_seq_eval,
                                           causal_masking=True))
        else:
            p_imagine_1 = jax.pmap(partial(sample_convSSM._imagine,
                                           model_seq=model_seq_eval,
                                           causal_masking=False))

        p_imagine_2 = jax.pmap(partial(sample_convSSM._imagine,
                                       model_seq=model_seq_eval,
                                       causal_masking=False))

        p_encode = jax.pmap(partial(sample_convSSM._encode,
                                    model_par=model_par_eval))
        p_decode = jax.pmap(partial(sample_convSSM._decode,
                                    model_par=model_par_eval))

    elif config.model in ["teco_transformer", "transformer", "performer"]:
        p_observe = jax.pmap(partial(sample_transformer._observe,
                                     model=model))
        if config.causal_masking:
            p_imagine_1 = jax.pmap(partial(sample_transformer._imagine,
                                           model=model,
                                           causal_masking=True), in_axes=(0, 0, 0, 0, None, 0))
        else:
            p_imagine_1 = jax.pmap(partial(sample_transformer._imagine,
                                           model=model,
                                           causal_masking=False), in_axes=(0, 0, 0, 0, None, 0))

        p_imagine_2 = jax.pmap(partial(sample_transformer._imagine,
                                       model=model,
                                       causal_masking=False), in_axes=(0, 0, 0, 0, None, 0))

        p_encode = jax.pmap(partial(sample_transformer._encode,
                                    model=model))
        p_decode = jax.pmap(partial(sample_transformer._decode,
                                    model=model))

    elif config.model in ["transformer_noVQ", "performer_noVQ"]:
        p_observe = jax.pmap(partial(sample_transformer_noVQ._observe,
                                     model=model))
        p_imagine = jax.pmap(partial(sample_transformer_noVQ._imagine,
                                       model=model), in_axes=(0, 0, 0, 0, None, 0))
        p_encode = None 
        p_decode = None 


    elif config.model in ["convS5_noVQ", "convLSTM_noVQ"]:
        model_par_eval = model(parallel=True, training=False)
        model_seq_eval = model(parallel=False, training=False)
        model = model(parallel=True, training=True)

        p_observe = jax.pmap(partial(sample_convSSM_noVQ._observe,
                                     model_par=model_par_eval))
        p_imagine = jax.pmap(partial(sample_convSSM_noVQ._imagine,
                                     model_seq=model_seq_eval))
        p_encode = None
        p_decode = None

    state, schedule_fn = init_model_state(init_rng, model, batch, config)
    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print('Restored from checkpoint')

    state = jax_utils.replicate(state)

    if config.model in ["transformer_noVQ", "performer_noVQ", "convS5_noVQ", 'convLSTM_noVQ']:
        folder_1 = osp.join(config.ckpt, 'samples_1')
        folder_2 = osp.join(config.ckpt, 'samples_2')
        folder_3 = osp.join(config.ckpt, 'samples_3')

        get_samples_noVQ(model, state,
                         test_loader, steps_per_eval,
                         config.action_conditioned_1, config.open_loop_ctx_1,
                         p_observe, p_imagine, p_encode, p_decode, config.eval_seq_len_1, folder_1)
        get_samples_noVQ(model, state,
                         test_loader, steps_per_eval,
                         config.action_conditioned_2, config.open_loop_ctx_2,
                         p_observe, p_imagine, p_encode, p_decode,
                         config.eval_seq_len_2, folder_2)
        get_samples_noVQ(model, state,
                         test_loader, steps_per_eval,
                         config.action_conditioned_3, config.open_loop_ctx_3,
                         p_observe, p_imagine, p_encode, p_decode,
                         config.eval_seq_len_3, folder_3)

    elif config.model in ["cwvae_noVQ"]:
        folder_1 = osp.join(config.ckpt, 'samples_1')
        folder_2 = osp.join(config.ckpt, 'samples_2')
        folder_3 = osp.join(config.ckpt, 'samples_3')
        get_samples_cwvae(state, test_loader, steps_per_eval,
                          config.open_loop_ctx_1, p_sample, config.eval_seq_len_1, folder_1)

        get_samples_cwvae(state, test_loader, steps_per_eval,
                          config.open_loop_ctx_2, p_sample, config.eval_seq_len_2, folder_2)

        get_samples_cwvae(state, test_loader, steps_per_eval,
                          config.open_loop_ctx_3, p_sample, config.eval_seq_len_3, folder_3)

    else:
        folder_1 = osp.join(config.ckpt, 'samples')
        folder_2 = osp.join(config.ckpt, 'samples')

        get_samples(model, state,
                    test_loader, steps_per_eval,
                    config.action_conditioned_1, config.open_loop_ctx_1, p_observe, p_imagine_1, p_encode, p_decode,
                    folder_1)

        get_samples(model, state,
                    test_loader, steps_per_eval,
                    config.action_conditioned_2, config.open_loop_ctx_2, p_observe, p_imagine_2, p_encode, p_decode,
                    folder_2)


def get_samples(model, state, test_loader, steps_per_eval, action_conditioned,
                open_loop_ctx, p_observe, p_imagine, p_encode, p_decode, folder):

    if action_conditioned:
        folder += '_action'
    folder += f'_{open_loop_ctx}'

    for batch_idx in range(steps_per_eval):
        batch = next(test_loader)

        if config.model in ["teco_convS5", "convS5", "teco_S5", "S5"]:
            predictions, real = sample_convSSM.sample(model, state, batch['video'], batch['actions'],
                                                      action_conditioned, open_loop_ctx, p_observe,
                                                      p_imagine,
                                                      p_encode, p_decode)

        elif config.model in ["teco_transformer", "transformer", "performer"]:
            predictions, real = sample_transformer.sample(model, state, batch['video'], batch['actions'],
                                                          action_conditioned, open_loop_ctx, p_observe, p_imagine,
                                                          p_encode, p_decode)

        save(batch_idx, predictions, real, folder, open_loop_ctx)
        print('Saved to', folder)


def get_samples_noVQ(model, state, test_loader, steps_per_eval, action_conditioned,
                     open_loop_ctx, p_observe, p_imagine, p_encode, p_decode, eval_seq_len, folder):

    if action_conditioned:
        folder += '_action'
    folder += f'_{open_loop_ctx}'

    for batch_idx in range(steps_per_eval):
        batch = next(test_loader)

        if config.model in ["convS5_noVQ", 'convLSTM_noVQ']:
            predictions, real = sample_convSSM_noVQ.sample(model, state, batch['video'], batch['actions'],
                                                           action_conditioned, open_loop_ctx, p_observe,
                                                           p_imagine, p_encode, p_decode, eval_seq_len)
        else:
            predictions, real = sample_transformer_noVQ.sample(model, state, batch['video'], batch['actions'],
                                                               action_conditioned, open_loop_ctx, p_observe, p_imagine,
                                                               p_encode, p_decode, eval_seq_len)

        save(batch_idx, predictions, real, folder, open_loop_ctx)
        print('Saved to', folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='/raid/moving_mnist_longer_eval')
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    args.run_id = args.output_dir

    if not osp.isabs(args.output_dir):
        if 'DATA_DIR' not in os.environ:
            os.environ['DATA_DIR'] = 'logs'
            print('DATA_DIR environment variable not set, default to logs/')
        root_folder = os.environ['DATA_DIR']
        args.output_dir = osp.join(root_folder, args.output_dir)

    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['viz_interval'] = 10
        config['save_interval'] = 10
        config['log_interval'] = 1
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')

    print(f'JAX local devices: {jax.local_device_count()}')

    main()
