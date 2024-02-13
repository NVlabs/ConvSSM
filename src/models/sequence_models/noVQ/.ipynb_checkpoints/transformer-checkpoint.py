from typing import Optional, Any
import optax
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp

from src.models.transformer.transformer import Transformer
from src.models.base import ResNetEncoder, ResNetDecoder


class TECO_NOVQ(nn.Module):
    config: Any
    dtype: Optional[Any] = jnp.float32

    @property
    def metrics(self):
        metrics = ['loss', 'mse_loss', 'l1_loss']
        return metrics

    def setup(self):
        config = self.config

        self.action_embeds = nn.Embed(config.action_dim + 1, config.action_embed_dim, dtype=self.dtype)

        # Posterior
        self.encoder =  ResNetEncoder(**config.encoder, dtype=self.dtype)
        ds = 1
        self.z_shape = tuple([d // ds for d in self.config.latent_shape])
        # Temporal Transformer
        z_kernel = [config.z_ds, config.z_ds]
        self.z_tfm_shape = tuple([d // config.z_ds for d in self.z_shape])
        self.z_proj = nn.Conv(config.z_tfm_kwargs['embed_dim'], z_kernel,
                              strides=z_kernel, use_bias=False, padding='VALID', dtype=self.dtype)

        self.sos = self.param('sos', nn.initializers.normal(stddev=0.02),
                              (*self.z_tfm_shape, config.z_tfm_kwargs['embed_dim'],), jnp.float32)
        self.z_tfm = Transformer(
            **config.z_tfm_kwargs, pos_embed_type='sinusoidal',
            shape=(config.seq_len, *self.z_tfm_shape),
            dtype=self.dtype
        )
        self.z_unproj = nn.ConvTranspose(config.embedding_dim, z_kernel, strides=z_kernel,
                                         padding='VALID', use_bias=False, dtype=self.dtype)

        # Decoder
        out_dim = self.config.channels
        self.decoder = ResNetDecoder(**config.decoder, image_size=0,
                                     out_dim=out_dim, dtype=self.dtype)

    def sample_timestep(self, z_embeddings, actions, cond, t):
        t -= self.config.n_cond
        actions = self.action_embeds(actions)
        deter = self.temporal_transformer(
            z_embeddings, actions, cond, deterministic=True
        )
        deter = deter[:, t]

        key = self.make_rng('sample')
        recon_logits = nn.tanh(self.decoder(deter))
        recon = recon_logits
        z_t = self.encoder(recon)
        return z_t, recon

    def _init_mask(self):
        n_per = np.prod(self.z_tfm_shape)
        mask = jnp.tril(jnp.ones((self.config.seq_len, self.config.seq_len), dtype=bool))
        mask = mask.repeat(n_per, axis=0).repeat(n_per, axis=1)
        return mask

    def encode(self, encodings):
        inp = encodings
        out = jax.vmap(self.encoder, 1, 1)(inp)
        return None, {'embeddings': out}

    def temporal_transformer(self, z_embeddings, actions, cond, deterministic=False):

        inp = z_embeddings

        actions = jnp.tile(actions[:, :, None, None], (1, 1, *inp.shape[2:4], 1))
        inp = jnp.concatenate([inp[:, :-1], actions[:, 1:]], axis=-1)
        inp = jax.vmap(self.z_proj, 1, 1)(inp)

        sos = jnp.tile(self.sos[None, None], (z_embeddings.shape[0], 1, 1, 1, 1))
        sos = jnp.asarray(sos, self.dtype)

        inp = jnp.concatenate([sos, inp], axis=1)
        deter = self.z_tfm(inp, mask=self._init_mask(), deterministic=deterministic)
        deter = deter[:, self.config.n_cond:]

        deter = jax.vmap(self.z_unproj, 1, 1)(deter)

        return deter

    def __call__(self, video, actions, deterministic=False):
        # video: BTCHW, actions: BT
        if not self.config.use_actions:
            if actions is None:
                actions = jnp.zeros(video.shape[:2], dtype=jnp.int32)
            else:
                actions = jnp.zeros_like(actions)

        if self.config.dropout_actions:
            dropout_actions = jax.random.bernoulli(self.make_rng('sample'), p=self.config.action_dropout_rate,
                                                   shape=(video.shape[0],))  # B
            if "minecraft" not in self.config.data_path:
                # Don't drop actions for minecraft
                actions = jnp.where(dropout_actions[:, None], self.config.action_mask_id, actions)
        else:
            dropout_actions = None

        actions = self.action_embeds(actions)
        encodings = video

        cond, vq_output = self.encode(encodings)
        z_embeddings =  vq_output['embeddings']

        deter = self.temporal_transformer(
            z_embeddings, actions, cond, deterministic=deterministic
        )

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
        recon_logits = nn.tanh(jax.vmap(self.decoder, 1, 1)(deter))

        mse_loss = 2*optax.l2_loss(recon_logits, labels) #optax puts a 0.5 in front automatically
        mse_loss = mse_loss.sum(axis=(-2, -1))
        mse_loss = mse_loss.mean()

        l1_loss = jnp.abs(recon_logits-labels)
        l1_loss = l1_loss.sum(axis=(-2, -1))
        l1_loss = l1_loss.mean()

        loss = self.config.loss_weight * mse_loss + (1-self.config.loss_weight) * l1_loss

        out = dict(loss=loss, mse_loss=mse_loss, l1_loss=l1_loss)
        return out

        
