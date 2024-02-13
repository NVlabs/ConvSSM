# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------


from typing import Optional, Any, Dict, Callable
import optax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from src.models.S5.diagonal_ssm import init_S5SSM
from src.models.base import ResNetEncoder, ResNetDecoder
from src.models.S5.layers import StackedLayers


class S5(nn.Module):
    config: Any
    vq_fns: Dict[str, Callable]
    vqvae: Any
    training: bool
    parallel: bool
    dtype: Optional[Any] = jnp.float32

    @property
    def metrics(self):
        metrics = ['loss']
        return metrics

    def setup(self):
        config = self.config

        # Sequence Model
        self.ssm = init_S5SSM(config.ssm['ssm_size'],
                              config.ssm['blocks'],
                              config.ssm['clip_eigs'],
                              config.d_model,
                              config.ssm['dt_min'],
                              config.ssm['dt_max'])
        self.sequence_model = StackedLayers(**self.config.seq_model,
                                            ssm=self.ssm,
                                            training=self.training,
                                            parallel=self.parallel)

        # initial_states = []
        bsz_device, _ = divmod(config.batch_size, jax.device_count())

        self.initial_states = jnp.zeros((bsz_device,
                                         config.seq_model['n_layers'],
                                         config.ssm['ssm_size'] // 2))

        self.action_embeds = nn.Embed(config.action_dim + 1, config.action_embed_dim, dtype=self.dtype)

        z_kernel = [config.z_ds, config.z_ds]
        self.z_proj = nn.Conv(config.d_model, z_kernel,
                              strides=z_kernel, use_bias=False, padding='VALID', dtype=self.dtype)
        self.z_unproj = nn.ConvTranspose(config.embedding_dim, z_kernel, strides=z_kernel,
                                         padding='VALID', use_bias=False, dtype=self.dtype)

        # Posterior
        self.encoder = ResNetEncoder(**config.encoder, dtype=self.dtype)

        # Decoder
        out_dim = self.vqvae.n_codes
        self.decoder = ResNetDecoder(**config.decoder, image_size=self.vqvae.latent_shape[0],
                                     out_dim=out_dim, dtype=self.dtype)

    def sample_timestep(self, encoding, initial_states, action, causal_masking):
        inp = self.encode(encoding)

        if causal_masking:
            inp = self.config.frame_mask_id * jnp.ones(inp.shape)

        action = self.action_embeds(action)
        action = jnp.tile(action[:, :, None, None], (1, 1, *inp.shape[2:4], 1))
        inp = jnp.concatenate([inp, action], axis=-1)
        inp = jax.vmap(self.z_proj, 1, 1)(inp)

        # inp is BTHWC, S5 model needs TBC
        inp = jnp.squeeze(inp, axis=(-2, -3))
        last_states, deter = jax.vmap(self.sequence_model)(inp, initial_states)
        deter = jnp.expand_dims(deter, (2, 3))

        deter = jax.vmap(self.z_unproj, 1, 1)(deter)

        recon_logits, recon = self.reconstruct(deter,
                                               sample=True,
                                               key=self.make_rng('sample'))
        return recon[:, 0], recon_logits, recon[:, 0], last_states

    def encode(self, encodings):
        embeddings = self.vq_fns['lookup'](encodings)

        out = jax.vmap(self.encoder, 1, 1)(embeddings)
        return out

    def condition(self, encodings, actions, initial_states=None,
                  mask_frames=False, drop_inds=None, num_input_frames=None):
        if initial_states is None:
            initial_states = self.initial_states

        # video: BTCHW, actions: BT
        inp = self.encode(encodings)

        if mask_frames:
            inp_embed = inp[:, :num_input_frames]
            mask_embed_shape = inp[:, num_input_frames:].shape
            if drop_inds is not None:
                inp = jnp.where(drop_inds[:, None, None, None, None],
                                inp,
                                jnp.concatenate((inp_embed,
                                                 self.config.frame_mask_id * jnp.ones(mask_embed_shape)),
                                                axis=1))
            else:
                inp = jnp.concatenate((inp_embed,
                                       self.config.frame_mask_id * jnp.ones(mask_embed_shape)), axis=1)

        # Combine inputs and actions
        actions = self.action_embeds(actions)
        actions = jnp.tile(actions[:, :, None, None], (1, 1, *inp.shape[2:4], 1))
        inp = jnp.concatenate([inp[:, :-1], actions[:, 1:]], axis=-1)
        inp = jax.vmap(self.z_proj, 1, 1)(inp)

        # inp is BTHWC, S5 model needs TBC
        inp = jnp.squeeze(inp, axis=(-2, -3))
        last_states, deter = jax.vmap(self.sequence_model)(inp, initial_states)
        deter = jnp.expand_dims(deter, (2, 3))
        deter = jax.vmap(self.z_unproj, 1, 1)(deter)

        return None, encodings, None, deter, last_states

    def reconstruct(self, deter, sample=False, key=None):
        recon_logits = jax.vmap(self.decoder, 1, 1)(deter)
        if sample:
            recon = random.categorical(key, recon_logits)
        else:
            recon = jnp.argmax(recon_logits, axis=-1)
        return recon_logits, recon

    def __call__(self, video, actions, deterministic=False):
        # video: BTHWC, actions: BT
        if not self.config.use_actions:
            if actions is None:
                actions = jnp.zeros(video.shape[:2], dtype=jnp.int32)
            else:
                actions = jnp.zeros_like(actions)

        if self.config.dropout_actions:
            dropout_actions = jax.random.bernoulli(self.make_rng('sample'), p=self.config.action_dropout_rate,
                                                   shape=(video.shape[0],))  # B
            actions = jnp.where(dropout_actions[:, None], self.config.action_mask_id, actions)
        else:
            dropout_actions = None

        _, encodings = self.vq_fns['encode'](video)

        if self.config.causal_masking:
            _, _, _, deter, _ = self.condition(encodings,
                                               actions,
                                               mask_frames=True,
                                               drop_inds=dropout_actions,
                                               num_input_frames=self.config.open_loop_ctx_1
                                               )
        else:
            _, _, _, deter, _ = self.condition(encodings, actions)

        encodings = encodings[:, self.config.n_cond:]
        labels = jax.nn.one_hot(encodings, num_classes=self.vqvae.n_codes)
        labels = labels * 0.99 + 0.01 / self.vqvae.n_codes  # Label smoothing

        if self.config.causal_masking:
            # Currently no support for droploss with causal masking
            recon_logits, _ = self.reconstruct(deter)

            recon_logits_out = recon_logits[:, self.config.open_loop_ctx_1 - 1:]
            labels_out = labels[:, self.config.open_loop_ctx_1 - 1:]

            recon_logits_1 = jnp.where(dropout_actions[:, None, None, None, None],
                                       labels_out,
                                       recon_logits_out)

            loss_1 = optax.softmax_cross_entropy(recon_logits_1, labels_out)
            loss_1 = loss_1.sum(axis=(-2, -1))
            loss_1 = loss_1.mean()

            recon_logits_2 = jnp.where(dropout_actions[:, None, None, None, None],
                                       recon_logits,
                                       labels)

            loss_2 = optax.softmax_cross_entropy(recon_logits_2, labels)
            loss_2 = loss_2.sum(axis=(-2, -1))
            loss_2 = loss_2.mean()

            loss = loss_1 + loss_2

        else:
            if self.config.drop_loss_rate is not None and self.config.drop_loss_rate > 0.0:
                n_sample = int((1 - self.config.drop_loss_rate) * deter.shape[1])
                n_sample = max(1, n_sample)
                idxs = jax.random.randint(self.make_rng('sample'),
                                          [n_sample],
                                          0, video.shape[1], dtype=jnp.int32)
            else:
                idxs = jnp.arange(deter.shape[1], dtype=jnp.int32)

            deter = deter[:, idxs]
            labels = labels[:, idxs]

            # Decoder loss
            recon_logits, _ = self.reconstruct(deter)
            recon_loss = optax.softmax_cross_entropy(recon_logits, labels)
            recon_loss = recon_loss.sum(axis=(-2, -1))
            recon_loss = recon_loss.mean()

            loss = recon_loss

        out = dict(loss=loss)
        return out


        
        
