# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------


from typing import Optional, Any
import optax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from src.models.convS5.conv_ops import VmapBasicConv
from src.models.convLSTM.ssm import init_ConvLSTM
from src.models.base import ResNetEncoder, ResNetDecoder
from src.models.convLSTM.layers import StackedLayers


def reshape_data(frames):
    # Make(seq_len, dev_bsz, H, W, in_dim)
    frames = frames.transpose(1, 0, 2, 3, 4)
    return frames


class CONVLSTM_NOVQ(nn.Module):
    config: Any
    training: bool
    parallel: bool
    dtype: Optional[Any] = jnp.float32

    @property
    def metrics(self):
        metrics = ['loss', 'mse_loss', 'l1_loss']
        return metrics

    def setup(self):
        config = self.config

        # Sequence Model
        self.ssm = init_ConvLSTM(config.d_model,
                                 config.ssm['ssm_size'],
                                 config.ssm['kernel_size'])
        self.sequence_model = StackedLayers(**self.config.seq_model,
                                            ssm=self.ssm,
                                            training=self.training,
                                            parallel=self.parallel)

        initial_states = []
        bsz_device, _ = divmod(config.batch_size, jax.device_count())
        for i in range(config.seq_model['n_layers']):
            initial_states.append(
                                    (np.zeros((bsz_device,
                                               config.latent_height,
                                               config.latent_width,
                                               config.ssm['ssm_size'])),
                                     np.zeros((bsz_device,
                                               config.latent_height,
                                               config.latent_width,
                                               config.ssm['ssm_size']))
                                     )
                                  )

        self.initial_states = initial_states

        self.action_embeds = nn.Embed(config.action_dim + 1, config.action_embed_dim, dtype=self.dtype)
        self.action_conv = VmapBasicConv(k_size=1,
                                         out_channels=config.d_model)

        # Encoder
        self.encoder = ResNetEncoder(**config.encoder, dtype=self.dtype)

        # Decoder
        out_dim = self.config.channels
        self.decoder = ResNetDecoder(**config.decoder, image_size=0,
                                     out_dim=out_dim, dtype=self.dtype)

    def sample_timestep(self, encoding, initial_states, action):
        inp = self.encode(encoding)

        action = self.action_embeds(action)
        action = jnp.tile(action[:, :, None, None], (1, 1, *inp.shape[2:4], 1))
        inp = jnp.concatenate([inp, action], axis=-1)

        # inp is BTHWC, convS5 model needs TBHWC
        inp = reshape_data(inp)
        inp = self.action_conv(inp)
        last_states, deter = self.sequence_model(inp, initial_states)
        deter = reshape_data(deter)  # Now BTHWC

        recon_logits, recon = self.reconstruct(deter)
        return recon, recon_logits, recon, last_states

    def encode(self, encodings):
        out = jax.vmap(self.encoder, 1, 1)(encodings)

        return out

    def condition(self, encodings, actions, initial_states=None):
        if initial_states is None:
            initial_states = self.initial_states

        # video: BTCHW, actions: BT
        inp = self.encode(encodings)

        # Combine inputs and actions
        actions = self.action_embeds(actions)
        actions = jnp.tile(actions[:, :, None, None], (1, 1, *inp.shape[2:4], 1))
        inp = jnp.concatenate([inp[:, :-1], actions[:, 1:]], axis=-1)

        # inp is BTHWC, convS5 model needs TBHWC
        inp = reshape_data(inp)
        inp = self.action_conv(inp)
        last_states, deter = self.sequence_model(inp, initial_states)
        deter = reshape_data(deter)  # swap back to BTHWC

        return None, encodings, None, deter, last_states

    def reconstruct(self, deter):
        recon_logits = jax.vmap(self.decoder, 1, 1)(deter)
        recon = nn.tanh(recon_logits)
        return recon, recon

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

        encodings = video

        _, _, _, deter, _ = self.condition(encodings, actions)

        labels = video[:, self.config.n_cond:]

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

        mse_loss = 2*optax.l2_loss(recon_logits, labels)  # optax puts a 0.5 in front automatically
        mse_loss = mse_loss.sum(axis=(-2, -1))
        mse_loss = mse_loss.mean()

        l1_loss = jnp.abs(recon_logits-labels)
        l1_loss = l1_loss.sum(axis=(-2, -1))
        l1_loss = l1_loss.mean()

        loss = self.config.loss_weight * mse_loss + (1-self.config.loss_weight) * l1_loss

        out = dict(loss=loss, mse_loss=mse_loss, l1_loss=l1_loss)
        return out
