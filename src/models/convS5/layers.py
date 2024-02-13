# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------


from flax import linen as nn


class SequenceLayer(nn.Module):
    """Defines a single layer with activation,
       layer/batch norm, pre/postnorm, dropout, etc"""
    ssm: nn.Module
    training: bool
    parallel: bool
    activation_fn: str = "gelu"
    dropout: float = 0.0
    use_norm: bool = True
    prenorm: bool = False
    per_layer_skip: bool = True
    num_groups: int = 32
    squeeze_excite: bool = False

    def setup(self):
        if self.activation_fn in ["relu"]:
            self.activation = nn.relu
        elif self.activation_fn in ["gelu"]:
            self.activation = nn.gelu
        elif self.activation_fn in ["swish"]:
            self.activation = nn.swish
        elif self.activation_fn in ["elu"]:
            self.activation = nn.elu

        self.seq = self.ssm(parallel=self.parallel,
                            activation=self.activation,
                            num_groups=self.num_groups,
                            squeeze_excite=self.squeeze_excite)

        if self.use_norm:
            self.norm = nn.LayerNorm()

        # TODO: Need to figure out dropout strategy, maybe drop whole channels?
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, u, x0):
        if self.per_layer_skip:
            skip = u
        else:
            skip = 0
        # Apply pre-norm if necessary
        if self.use_norm:
            if self.prenorm:
                u = self.norm(u)
        x_L, u = self.seq(u, x0)
        u = self.drop(u)
        u = skip + u
        if self.use_norm:
            if not self.prenorm:
                u = self.norm(u)
        return x_L, u


class StackedLayers(nn.Module):
    """Stacks S5 layers
     output: outputs LxbszxH_uxW_uxU sequence of outputs and
             a list containing the last state of each layer"""
    ssm: nn.Module
    n_layers: int
    training: bool
    parallel: bool
    layer_activation: str = "gelu"
    dropout: float = 0.0
    use_norm: bool = False
    prenorm: bool = False
    skip_connections: bool = False
    per_layer_skip: bool = True
    num_groups: int = 32
    squeeze_excite: bool = False

    def setup(self):

        self.layers = [
            SequenceLayer(
                ssm=self.ssm,
                activation_fn=self.layer_activation,
                dropout=self.dropout,
                training=self.training,
                parallel=self.parallel,
                use_norm=self.use_norm,
                prenorm=self.prenorm,
                per_layer_skip=self.per_layer_skip,
                num_groups=self.num_groups,
                squeeze_excite=self.squeeze_excite
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, u, initial_states):
        # u is shape (L, bsz, d_in, im_H, im_W)
        # x0s is a list of initial arrays each of shape (bsz, d_model, im_H, im_W)
        last_states = []
        for i in range(len(self.layers)):
            if self.skip_connections:
                if i == 3:
                    layer9_in = u
                elif i == 6:
                    layer12_in = u

                if i == 8:
                    u = u + layer9_in
                elif i == 11:
                    u = u + layer12_in

            x_L, u = self.layers[i](u, initial_states[i])
            last_states.append(x_L)  # keep last state of each layer
        return last_states, u
