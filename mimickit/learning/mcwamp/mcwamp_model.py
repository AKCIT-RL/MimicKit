import torch

import learning.amp_model as amp_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class MCWAMPModel(amp_model.AMPModel):
    """AMP model with two independent critics (goal and aux streams).

    eval_critic returns [..., 2]: column 0 = goal (task) value,
    column 1 = aux (style) value.
    """

    def _build_critic(self, config, env):
        net_name = config["critic_net"]
        input_dict = self._build_critic_input_dict(env)

        self._critic_layers, _ = net_builder.build_net(net_name, input_dict,
                                                       activation=self._activation)
        layers_out_size = torch_util.calc_layers_out_size(self._critic_layers)
        self._critic_out = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.zeros_(self._critic_out.bias)

        self._aux_critic_layers, _ = net_builder.build_net(net_name, input_dict,
                                                           activation=self._activation)
        aux_layers_out_size = torch_util.calc_layers_out_size(self._aux_critic_layers)
        self._aux_critic_out = torch.nn.Linear(aux_layers_out_size, 1)
        torch.nn.init.zeros_(self._aux_critic_out.bias)
        return

    def eval_critic(self, obs):
        goal_val = self._critic_out(self._critic_layers(obs))
        aux_val = self._aux_critic_out(self._aux_critic_layers(obs))
        val = torch.cat([goal_val, aux_val], dim=-1)
        return val

    def get_critic_params(self):
        params = list(self._critic_layers.parameters()) + list(self._critic_out.parameters()) \
            + list(self._aux_critic_layers.parameters()) + list(self._aux_critic_out.parameters())
        return params
