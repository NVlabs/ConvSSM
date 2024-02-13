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

from typing import Optional, Any, Dict, Callable
import optax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from src.models.convS5.conv_ops import VmapBasicConv
from src.models.convS5.diagonal_ssm import init_ConvS5SSM
from src.models.base import ResNetEncoder, Codebook, ResNetDecoder
from src.models.convS5.layers import StackedLayers
from src.models.transformer.maskgit import MaskGit


def reshape_data(frames):
    # Make(seq_len, dev_bsz, H, W, in_dim)
    frames = frames.transpose(1, 0, 2, 3, 4)
    return frames


class TECO_CONVS5(nn.Module):
    config: Any
    vq_fns: Dict[str, Callable]
    vqvae: Any
    training: bool
    parallel: bool
    dtype: Optional[Any] = jnp.float32

    @property
    def metrics(self):
        metrics = ['loss', 'recon_loss', 'trans_loss', 'codebook_loss',
                   'commitment_loss', 'perplexity']
        return metrics

    def setup(self):
        config = self.config

        # Sequence Model
        self.ssm = init_ConvS5SSM(config.ssm['ssm_size'],
                                  config.ssm['blocks'],
                                  config.ssm['clip_eigs'],
                                  config.d_model,
                                  config.ssm['B_kernel_size'],
                                  config.ssm['C_kernel_size'],
                                  config.ssm['D_kernel_size'],
                                  config.ssm['dt_min'],
                                  config.ssm['dt_max'],
                                  config.ssm['C_D_config'])
        self.sequence_model = StackedLayers(**self.config.seq_model,
                                            ssm=self.ssm,
                                            training=self.training,
                                            parallel=self.parallel)

        initial_states = []
        bsz_device, _ = divmod(config.batch_size, jax.device_count())
        for i in range(config.seq_model['n_layers']):
            initial_states.append(jnp.zeros((bsz_device,
                                             config.latent_height,
                                             config.latent_width,
                                             config.ssm['ssm_size'] // 2))
                                  )

        self.initial_states = initial_states

        self.action_embeds = nn.Embed(config.action_dim + 1, config.action_embed_dim, dtype=self.dtype)
        self.action_conv = VmapBasicConv(k_size=1,
                                         out_channels=config.d_model)

        # Posterior
        self.sos_post = self.param('sos_post', nn.initializers.normal(stddev=0.02),
                                   (*self.vqvae.latent_shape, self.vqvae.embedding_dim), jnp.float32)
        self.encoder = nn.Sequential([
            ResNetEncoder(**config.encoder, dtype=self.dtype),
            nn.Dense(config.embedding_dim, dtype=self.dtype)
        ])
        ds = 2 ** (len(config.encoder['depths']) - 1)
        self.z_shape = tuple([d // ds for d in self.vqvae.latent_shape])
        self.codebook = Codebook(**self.config.codebook, embedding_dim=config.embedding_dim,
                                 dtype=self.dtype)

        # Dynamics Prior
        self.z_git = nn.vmap(
                MaskGit,
                in_axes=(1, 1, None), out_axes=1,
                variable_axes={'params': None},
                split_rngs={'params': False, 'sample': True, 'dropout': True}
        )(shape=self.z_shape, vocab_size=self.codebook.n_codes,
          **config.z_git, dtype=self.dtype)

        # Decoder
        out_dim = self.vqvae.n_codes
        self.decoder = ResNetDecoder(**config.decoder, image_size=self.vqvae.latent_shape[0],
                                     out_dim=out_dim, dtype=self.dtype)

    def sample_timestep(self, z_embedding, initial_states, action, causal_masking):
        inp = z_embedding

        if causal_masking:
            inp = self.config.frame_mask_id * jnp.ones(inp.shape)

        action = self.action_embeds(action)
        action = jnp.tile(action[:, :, None, None], (1, 1, *inp.shape[2:4], 1))
        inp = jnp.concatenate([inp, action], axis=-1)

        # inp is BTHWC, convS5 model needs TBHWC
        inp = reshape_data(inp)
        inp = self.action_conv(inp)
        last_states, deter = self.sequence_model(inp, initial_states)
        deter = reshape_data(deter)  # Now BTHWC

        deter = deter[:, 0]
        sample = self.z_git.sample(z_embedding.shape[0], self.config.T_draft,
                                   self.config.T_revise, self.config.M,
                                   cond=deter)
        z_t = self.codebook(None, encoding_indices=sample)

        recon = jnp.argmax(self.decoder(deter, z_t), axis=-1)
        return z_t, None, recon, last_states

    def encode(self, encodings):
        embeddings = self.vq_fns['lookup'](encodings)
        sos = jnp.tile(self.sos_post[None, None], (embeddings.shape[0], 1, 1, 1, 1))
        sos = jnp.asarray(sos, self.dtype)
        embeddings = jnp.concatenate([sos, embeddings], axis=1)
        inp = jnp.concatenate([embeddings[:, :-1], embeddings[:, 1:]], axis=-1)

        out = jax.vmap(self.encoder, 1, 1)(inp)
        vq_output = self.codebook(out)
        vq_embeddings, vq_encodings = vq_output['embeddings'], vq_output['encodings']

        vq_output = {
            'embeddings': vq_embeddings[:, self.config.n_cond:],
            'encodings': vq_encodings[:, self.config.n_cond:],
            'commitment_loss': vq_output['commitment_loss'],
            'codebook_loss': vq_output['codebook_loss'],
            'perplexity': vq_output['perplexity'],
        }

        return out[:, :self.config.n_cond], vq_output

    def condition(self, encodings, actions, initial_states=None,
                  mask_frames=False, drop_inds=None, num_input_frames=None):
        if initial_states is None:
            initial_states = self.initial_states

        # video: BTCHW, actions: BT
        cond, vq_output = self.encode(encodings)
        z_embeddings, z_codes = vq_output['embeddings'], vq_output['encodings']
        inp = jnp.concatenate([cond, z_embeddings], axis=1)

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

        # inp is BTHWC, convS5 model needs TBHWC
        inp = reshape_data(inp)
        inp = self.action_conv(inp)
        last_states, deter = self.sequence_model(inp, initial_states)
        deter = reshape_data(deter)  # swap back to BTHWC

        return vq_output, z_embeddings, z_codes, deter, last_states

    def reconstruct(self, deter, z_embeddings):
        recon_logits = jax.vmap(self.decoder, 1, 1)(deter, z_embeddings)
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
            vq_output, z_embeddings, z_codes, deter, _ = self.condition(encodings,
                                                                        actions,
                                                                        mask_frames=True,
                                                                        drop_inds=dropout_actions,
                                                                        num_input_frames=self.config.open_loop_ctx_1
                                                                        )
        else:
            vq_output, z_embeddings, z_codes, deter, _ = self.condition(encodings,
                                                                        actions)

        encodings = encodings[:, self.config.n_cond:]
        labels = jax.nn.one_hot(encodings, num_classes=self.vqvae.n_codes)
        labels = labels * 0.99 + 0.01 / self.vqvae.n_codes  # Label smoothing

        if self.config.drop_loss_rate is not None and self.config.drop_loss_rate > 0.0:
            n_sample = int((1 - self.config.drop_loss_rate) * deter.shape[1])
            n_sample = max(1, n_sample)
            idxs = jax.random.randint(self.make_rng('sample'),
                                      [n_sample],
                                      0, video.shape[1], dtype=jnp.int32)
        else:
            idxs = jnp.arange(deter.shape[1], dtype=jnp.int32)

        deter = deter[:, idxs]
        z_embeddings = z_embeddings[:, idxs]
        z_codes = z_codes[:, idxs]
        labels = labels[:, idxs]

        # Dynamics Prior loss
        z_logits, z_labels, z_mask = self.z_git(z_codes, deter, deterministic)

        if self.config.causal_masking:
            act_cond_mask = idxs >= self.config.open_loop_ctx_1 - 1
            act_cond_mask = act_cond_mask[None, ..., None, None, None]
            z_mask_act = z_mask * act_cond_mask

            z_logits_1 = jnp.where(dropout_actions[:, None, None, None, None],
                                   z_labels,
                                   z_logits)

            trans_loss_1 = optax.softmax_cross_entropy(z_logits_1, z_labels)
            trans_loss_1 = (trans_loss_1 * z_mask_act).sum() / z_mask_act.sum()
            trans_loss_1 = trans_loss_1 * np.prod(self.z_shape)

            z_logits_2 = jnp.where(dropout_actions[:, None, None, None, None],
                                   z_logits,
                                   z_labels)

            trans_loss_2 = optax.softmax_cross_entropy(z_logits_2, z_labels)
            trans_loss_2 = (trans_loss_2 * z_mask_act).sum() / z_mask_act.sum()
            trans_loss_2 = trans_loss_2 * np.prod(self.z_shape)

            trans_loss = trans_loss_1 + trans_loss_2

        else:
            trans_loss = optax.softmax_cross_entropy(z_logits, z_labels)
            trans_loss = (trans_loss * z_mask).sum() / z_mask.sum()
            trans_loss = trans_loss * np.prod(self.z_shape)

        # Decoder loss
        recon_logits, _ = self.reconstruct(deter, z_embeddings)
        recon_loss = optax.softmax_cross_entropy(recon_logits, labels)
        recon_loss = recon_loss.sum(axis=(-2, -1))
        recon_loss = recon_loss.mean()

        loss = recon_loss + trans_loss + vq_output['commitment_loss'] + \
            vq_output['codebook_loss']

        out = dict(loss=loss, recon_loss=recon_loss, trans_loss=trans_loss,
                   commitment_loss=vq_output['commitment_loss'],
                   codebook_loss=vq_output['codebook_loss'],
                   perplexity=vq_output['perplexity'])
        return out


        
        
