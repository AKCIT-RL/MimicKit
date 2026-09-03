"""Encoder-decoder MCWAMP model (arXiv:2511.03996, Frente F).

Actor input is the measurable obs published by TaskSoccerEnv in
measurable_obs mode: [current frame | H past frames], one frame being
projected gravity + base angular velocity + joint offsets + joint velocities
+ previous action + the 12-dim task block. The encoder compresses the flat
history into a latent that is concatenated to the current frame before the
actor MLP; a decoder reconstructs the true ball state from the latent
(training-only auxiliary loss, MCWAMPEncAgent). The critics are sized from
the env's privileged critic_obs, which in this mode has a DIFFERENT shape
from the actor obs.
"""

import types

import torch

import learning.mcwamp_model as mcwamp_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class MCWAMPEncModel(mcwamp_model.MCWAMPModel):

    def __init__(self, config, env):
        self._frame_dim = int(env.get_measurable_frame_dim())
        self._hist_steps = int(env.get_measurable_hist_steps())
        self._latent_dim = int(config.get("enc_latent_dim", 64))
        self._recon_dim = int(env.get_recon_tar_size())
        super().__init__(config, env)
        return

    def _build_actor(self, config, env):
        enc_net = config.get("enc_net", "fc_2layers_256units")
        hist_spec = types.SimpleNamespace(shape=(self._hist_steps * self._frame_dim,))
        self._enc_layers, _ = net_builder.build_net(enc_net, {"hist": hist_spec},
                                                    activation=self._activation)
        enc_out_size = torch_util.calc_layers_out_size(self._enc_layers)
        self._enc_out = torch.nn.Linear(enc_out_size, self._latent_dim)
        torch.nn.init.zeros_(self._enc_out.bias)

        dec_net = config.get("dec_net", enc_net)
        latent_spec = types.SimpleNamespace(shape=(self._latent_dim,))
        self._dec_layers, _ = net_builder.build_net(dec_net, {"latent": latent_spec},
                                                    activation=self._activation)
        dec_out_size = torch_util.calc_layers_out_size(self._dec_layers)
        self._dec_out = torch.nn.Linear(dec_out_size, self._recon_dim)
        torch.nn.init.zeros_(self._dec_out.bias)

        net_name = config["actor_net"]
        actor_spec = types.SimpleNamespace(shape=(self._frame_dim + self._latent_dim,))
        self._actor_layers, _ = net_builder.build_net(net_name, {"obs": actor_spec},
                                                      activation=self._activation)
        self._action_dist = self._build_action_distribution(config, env, self._actor_layers)
        return

    def _split_obs(self, obs):
        frame = obs[..., :self._frame_dim]
        hist = obs[..., self._frame_dim:]
        return frame, hist

    def eval_latent(self, obs):
        _, hist = self._split_obs(obs)
        return self._enc_out(self._enc_layers(hist))

    def eval_actor(self, obs):
        frame, _ = self._split_obs(obs)
        z = self.eval_latent(obs)
        h = self._actor_layers(torch.cat([frame, z], dim=-1))
        return self._action_dist(h)

    def eval_recon(self, obs):
        z = self.eval_latent(obs)
        return self._dec_out(self._dec_layers(z))

    def get_actor_params(self):
        params = super().get_actor_params() \
            + list(self._enc_layers.parameters()) + list(self._enc_out.parameters()) \
            + list(self._dec_layers.parameters()) + list(self._dec_out.parameters())
        return params

    def _build_critic_input_dict(self, env):
        return {"obs": env.get_critic_obs_space()}
