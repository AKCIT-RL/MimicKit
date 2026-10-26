import torch

import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class KickDisc(torch.nn.Module):
    """Second WAMP critic that only sees kicks (doc 25 phase C(c), goal M13).

    Demo samples: disc obs windows of the dataset kick clips around the strike
    (env.fetch_kick_disc_obs_demo). Agent samples: steps with a foot near the
    ball (info["kick_near"]). Same net, loss and reward mapping as the main
    critic; its own running score stats for the reward normalization.
    """

    def __init__(self, model_config, env):
        super().__init__()
        activation = getattr(torch.nn, model_config.get("activation", "ReLU"))
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        self._layers, _ = net_builder.build_net(model_config["disc_net"], input_dict, activation=activation)
        self._logits = torch.nn.Linear(torch_util.calc_layers_out_size(self._layers), 1)
        torch.nn.init.uniform_(self._logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._logits.bias)
        self.register_buffer("score_mean", torch.zeros([1], dtype=torch.float32))
        self.register_buffer("score_var", torch.ones([1], dtype=torch.float32))
        return

    def forward(self, disc_obs):
        return self._logits(self._layers(disc_obs))

    def get_logit_weights(self):
        return torch.flatten(self._logits.weight)
