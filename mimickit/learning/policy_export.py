"""Export a trained MimicKit policy as a standalone TorchScript module.

The deployment target is booster_deploy, whose consumer contract is one line
(tasks/locomotion/locomotion.py):

    self._model = torch.jit.load(policy_path, map_location="cpu")
    action = self._model(obs).squeeze(0)

IsaacLab's rsl_rl exporter satisfies that with `forward(x) = actor(normalizer(x))`
and a `@torch.jit.export reset()`. This is the MimicKit equivalent, with one
deliberate difference explained below.

WHAT THE EXPORTED MODULE COMPUTES

The agent's deterministic (TEST-mode) action path is four steps, spread over
three files:

    norm_obs = _obs_norm.normalize(obs)                 ppo_agent._decide_action
    norm_a   = _model.eval_actor(norm_obs).mode         == _mean_net(actor_layers(.))
    a        = _a_norm.unnormalize(norm_a)              ppo_agent._decide_action
    a        = clamp(a, bound_low, bound_high)          char_env._apply_action

`a` is then handed to the engine as a JOINT POSITION TARGET (set_cmd, control
mode "pos"). All four steps are baked into the exported module, so its output is
directly what the PD controller should track.

That last step matters and is easy to miss: rsl_rl leaves the action scaling to
the environment (`default_joint_pos + action * action_scale`), so an rsl_rl
export is only half a policy. MimicKit's action normalizer is derived from the
asset's joint limits, and its clip bounds from those again - properties of the
robot model, not choices a consumer should be asked to re-declare. Leaking them
into the deployment code is how a policy silently runs at the wrong scale. They
are baked in here instead. A consumer must therefore NOT add a default pose or
apply an action scale to this module's output.

WHY THE CLIP NEEDS NO ASSET FILE

char_env._apply_action clips to the action space bounds, which live in the
environment, not in the checkpoint. They are recoverable exactly. base_agent.
_build_action_normalizer builds the normalizer from that same Box space:

    a_mean = 0.5 * (high + low)        =>  a_mean - a_std == low
    a_std  = 0.5 * (high - low)            a_mean + a_std == high

So clipping to [low, high] in raw action space is identically clipping to
[-1, 1] in normalized space, before unnormalizing. This is an algebraic
identity, not an approximation, and it holds for any Box action space - zero
centred or not. The exported module uses the normalized form, which is why a
checkpoint alone is enough to export: no asset, no env, no simulator.

JOINT ORDER

The output is in the asset's DOF order. For the T1 that is the MJCF order,
which scripts/prepare_t1_asset.py asserts is identical to T1_23DOF_CFG's order
in booster_deploy/booster_deploy/robots/booster.py - so no remapping is needed
on that side. For any other embodiment, check before wiring it up.
"""

import types

import torch

import learning.distribution_gaussian_diag as distribution_gaussian_diag
import learning.nets.net_builder as net_builder
import learning.normalizer as normalizer
import util.torch_util as torch_util

OBS_NORM_PREFIX = "_obs_norm."
A_NORM_PREFIX = "_a_norm."
ACTOR_LAYERS_PREFIX = "_model._actor_layers."
MEAN_NET_PREFIX = "_model._action_dist._mean_net."
ENC_LAYERS_PREFIX = "_model._enc_layers."
ENC_OUT_PREFIX = "_model._enc_out."

# base_agent._build_normalizers hardcodes clip=10.0 on the observation
# normalizer. It is not stored in the checkpoint, so it is mirrored here.
OBS_NORM_CLIP = 10.0


class ExportedPolicy(torch.nn.Module):
    """Deterministic policy: raw observation in, joint position targets out.

    Deliberately written as inlined tensor arithmetic rather than by calling
    Normalizer/DistributionGaussianDiag. Two reasons: TorchScript cannot script
    a module whose forward returns a plain Python object (the distribution), and
    an independent implementation is what makes the equivalence test in
    tests/unit/test_policy_export.py able to fail.
    """

    def __init__(self, obs_mean, obs_std, obs_clip, actor_layers, mean_net,
                 a_mean, a_std, clip_action=True):
        super().__init__()

        self.register_buffer("_obs_mean", obs_mean.detach().clone())
        self.register_buffer("_obs_std", obs_std.detach().clone())
        self.register_buffer("_a_mean", a_mean.detach().clone())
        self.register_buffer("_a_std", a_std.detach().clone())

        self._actor_layers = actor_layers
        self._mean_net = mean_net
        self._obs_clip = float(obs_clip)
        self._clip_action = bool(clip_action)
        return

    def forward(self, obs):
        norm_obs = (obs - self._obs_mean) / self._obs_std
        norm_obs = torch.clamp(norm_obs, -self._obs_clip, self._obs_clip)

        norm_a = self._mean_net(self._actor_layers(norm_obs))

        if (self._clip_action):
            # char_env._apply_action, moved into normalized space (see header)
            norm_a = torch.clamp(norm_a, -1.0, 1.0)

        return norm_a * self._a_std + self._a_mean

    @torch.jit.export
    def reset(self):
        # booster_deploy's Policy contract calls reset() between episodes. This
        # policy is memoryless - it consumes one frame, keeps no history - so
        # there is nothing to clear. Present so the exported module satisfies
        # the interface directly, matching IsaacLab's exporter.
        pass


class ExportedEncPolicy(torch.nn.Module):
    """Deterministic encoder-decoder policy (Frente F), STATELESS by design.

    forward(obs) where obs = [current frame | H past frames, oldest first]
    flattened, in RAW units. The consumer owns the frame ring buffer (newest
    last) and, after a reset, refills it with the first measured frame
    repeated - exactly like the training env (no zero transient). The encoder
    and the split are baked in; the decoder is training-only and not exported.
    """

    def __init__(self, obs_mean, obs_std, obs_clip, enc_layers, enc_out,
                 actor_layers, mean_net, a_mean, a_std, frame_dim,
                 clip_action=True):
        super().__init__()

        self.register_buffer("_obs_mean", obs_mean.detach().clone())
        self.register_buffer("_obs_std", obs_std.detach().clone())
        self.register_buffer("_a_mean", a_mean.detach().clone())
        self.register_buffer("_a_std", a_std.detach().clone())

        self._enc_layers = enc_layers
        self._enc_out = enc_out
        self._actor_layers = actor_layers
        self._mean_net = mean_net
        self._frame_dim = int(frame_dim)
        self._obs_clip = float(obs_clip)
        self._clip_action = bool(clip_action)
        return

    def forward(self, obs):
        norm_obs = (obs - self._obs_mean) / self._obs_std
        norm_obs = torch.clamp(norm_obs, -self._obs_clip, self._obs_clip)

        frame = norm_obs[..., :self._frame_dim]
        hist = norm_obs[..., self._frame_dim:]
        z = self._enc_out(self._enc_layers(hist))
        norm_a = self._mean_net(self._actor_layers(torch.cat([frame, z], dim=-1)))

        if (self._clip_action):
            norm_a = torch.clamp(norm_a, -1.0, 1.0)

        return norm_a * self._a_std + self._a_mean

    @torch.jit.export
    def reset(self):
        # stateless on purpose: the frame history is an INPUT, owned by the
        # consumer, so there is no hidden state to clear
        pass


def load_checkpoint(path, device="cpu"):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    for key in (OBS_NORM_PREFIX + "_mean", A_NORM_PREFIX + "_mean",
                ACTOR_LAYERS_PREFIX + "0.weight", MEAN_NET_PREFIX + "weight"):
        if (key not in state_dict):
            raise KeyError("'{}' is missing from {}; this does not look like a "
                           "PPO-family MimicKit checkpoint".format(key, path))
    return state_dict


def get_sizes(state_dict):
    obs_size = state_dict[OBS_NORM_PREFIX + "_mean"].shape[0]
    a_size = state_dict[A_NORM_PREFIX + "_mean"].shape[0]
    return int(obs_size), int(a_size)


def has_encoder(state_dict):
    return (ENC_LAYERS_PREFIX + "0.weight") in state_dict


def get_enc_sizes(state_dict):
    """(frame_dim, hist_size, latent_dim), all derived from the weights."""
    obs_size, _ = get_sizes(state_dict)
    hist_size = int(state_dict[ENC_LAYERS_PREFIX + "0.weight"].shape[1])
    latent_dim = int(state_dict[ENC_OUT_PREFIX + "weight"].shape[0])
    frame_dim = obs_size - hist_size
    if (frame_dim <= 0 or hist_size % frame_dim != 0):
        raise ValueError("encoder input {} is not a whole number of {}-dim "
                         "frames".format(hist_size, frame_dim))
    return frame_dim, hist_size, latent_dim


def _sub_state_dict(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _sanitize_for_script(module):
    """Cast numpy integer layer attributes to Python ints.

    MimicKit's net builders size the first layer with
    `np.sum([np.prod(s.shape) for ...])`, which returns a numpy.int64. That is
    harmless in eager mode, but torch.jit.script rejects it:

        TypeError: 'numpy.int64' object in attribute 'Linear.in_features'
                   is not a valid constant.

    Fixed here rather than in net_builder so that exporting cannot perturb
    training. Weights are untouched - only the bookkeeping attributes change.
    """
    for m in module.modules():
        for attr in ("in_features", "out_features"):
            value = getattr(m, attr, None)
            if (value is not None and not isinstance(value, int)):
                setattr(m, attr, int(value))
    return module


def build_actor(state_dict, actor_net, device="cpu", activation=torch.nn.ReLU):
    """Rebuild the actor with MimicKit's own builders and load the weights.

    Deliberately calls net_builder.build_net and DistributionGaussianDiagBuilder
    rather than hand-rolling Linear layers: an architecture change upstream then
    surfaces as a load_state_dict error here instead of as a wrong export. The
    strict load is the drift guard, so do not relax it.

    build_net only reads `.shape` off each entry of its input dict, so a stub
    stands in for the gym Box the environment would normally supply - that is
    the whole reason no simulator is needed.
    """
    obs_size, a_size = get_sizes(state_dict)

    obs_spec = types.SimpleNamespace(shape=(obs_size,))
    actor_layers, _ = net_builder.build_net(actor_net, {"obs": obs_spec},
                                            activation=activation)

    in_size = torch_util.calc_layers_out_size(actor_layers)
    # std_type/init_std/init_output_scale only set initial values, all of which
    # the state_dict load overwrites; FIXED avoids building an unused logstd net
    dist_builder = distribution_gaussian_diag.DistributionGaussianDiagBuilder(
        in_size, a_size,
        std_type=distribution_gaussian_diag.StdType.FIXED,
        init_std=1.0, init_output_scale=0.01)

    actor_layers.load_state_dict(_sub_state_dict(state_dict, ACTOR_LAYERS_PREFIX))
    dist_builder._mean_net.load_state_dict(_sub_state_dict(state_dict, MEAN_NET_PREFIX))

    _sanitize_for_script(actor_layers)
    _sanitize_for_script(dist_builder)

    actor_layers.to(device).eval()
    dist_builder.to(device).eval()
    return actor_layers, dist_builder


def build_enc_actor(state_dict, actor_net, enc_net, device="cpu",
                    activation=torch.nn.ReLU):
    """Rebuild encoder + actor with MimicKit's own builders (strict load)."""
    _, a_size = get_sizes(state_dict)
    frame_dim, hist_size, latent_dim = get_enc_sizes(state_dict)

    hist_spec = types.SimpleNamespace(shape=(hist_size,))
    enc_layers, _ = net_builder.build_net(enc_net, {"hist": hist_spec},
                                          activation=activation)
    enc_out = torch.nn.Linear(torch_util.calc_layers_out_size(enc_layers), latent_dim)

    actor_spec = types.SimpleNamespace(shape=(frame_dim + latent_dim,))
    actor_layers, _ = net_builder.build_net(actor_net, {"obs": actor_spec},
                                            activation=activation)

    in_size = torch_util.calc_layers_out_size(actor_layers)
    dist_builder = distribution_gaussian_diag.DistributionGaussianDiagBuilder(
        in_size, a_size,
        std_type=distribution_gaussian_diag.StdType.FIXED,
        init_std=1.0, init_output_scale=0.01)

    enc_layers.load_state_dict(_sub_state_dict(state_dict, ENC_LAYERS_PREFIX))
    enc_out.load_state_dict(_sub_state_dict(state_dict, ENC_OUT_PREFIX))
    actor_layers.load_state_dict(_sub_state_dict(state_dict, ACTOR_LAYERS_PREFIX))
    dist_builder._mean_net.load_state_dict(_sub_state_dict(state_dict, MEAN_NET_PREFIX))

    for module in (enc_layers, enc_out, actor_layers, dist_builder):
        _sanitize_for_script(module)
        module.to(device).eval()
    return enc_layers, enc_out, actor_layers, dist_builder


def build_normalizers(state_dict, device="cpu", obs_clip=OBS_NORM_CLIP):
    """Rebuild the two normalizers as real Normalizer objects."""
    obs_size, a_size = get_sizes(state_dict)

    obs_norm = normalizer.Normalizer([obs_size], device=device, clip=obs_clip)
    obs_norm.load_state_dict(_sub_state_dict(state_dict, OBS_NORM_PREFIX))

    a_norm = normalizer.Normalizer([a_size], device=device)
    a_norm.load_state_dict(_sub_state_dict(state_dict, A_NORM_PREFIX))
    return obs_norm, a_norm


def build_exported_policy(state_dict, actor_net, device="cpu",
                          clip_action=True, obs_clip=OBS_NORM_CLIP):
    """The module that gets scripted and shipped."""
    actor_layers, dist_builder = build_actor(state_dict, actor_net, device=device)

    policy = ExportedPolicy(
        obs_mean=state_dict[OBS_NORM_PREFIX + "_mean"],
        obs_std=state_dict[OBS_NORM_PREFIX + "_std"],
        obs_clip=obs_clip,
        actor_layers=actor_layers,
        mean_net=dist_builder._mean_net,
        a_mean=state_dict[A_NORM_PREFIX + "_mean"],
        a_std=state_dict[A_NORM_PREFIX + "_std"],
        clip_action=clip_action)
    policy.to(device).eval()
    return policy


def build_reference_policy(state_dict, actor_net, device="cpu",
                           clip_action=True, obs_clip=OBS_NORM_CLIP):
    """The same policy assembled out of MimicKit's own classes.

    This is the reference side of the equivalence test, and it is written to
    follow ppo_agent._decide_action and char_env._apply_action step for step -
    including clipping in RAW action space, where the environment does it, not
    in the normalized space the exported module uses. The two agreeing is what
    validates the identity argued in this module's header.
    """
    obs_norm, a_norm = build_normalizers(state_dict, device=device, obs_clip=obs_clip)
    actor_layers, dist_builder = build_actor(state_dict, actor_net, device=device)

    bound_low = a_norm.get_mean() - a_norm.get_std()
    bound_high = a_norm.get_mean() + a_norm.get_std()

    def policy(obs):
        with torch.no_grad():
            norm_obs = obs_norm.normalize(obs)
            a_dist = dist_builder(actor_layers(norm_obs))
            a = a_norm.unnormalize(a_dist.mode)
            if (clip_action):
                a = torch.minimum(torch.maximum(a, bound_low), bound_high)
        return a

    return policy


def build_exported_enc_policy(state_dict, actor_net, enc_net, device="cpu",
                              clip_action=True, obs_clip=OBS_NORM_CLIP):
    """The encoder-policy module that gets scripted and shipped."""
    enc_layers, enc_out, actor_layers, dist_builder = build_enc_actor(
        state_dict, actor_net, enc_net, device=device)
    frame_dim, _, _ = get_enc_sizes(state_dict)

    policy = ExportedEncPolicy(
        obs_mean=state_dict[OBS_NORM_PREFIX + "_mean"],
        obs_std=state_dict[OBS_NORM_PREFIX + "_std"],
        obs_clip=obs_clip,
        enc_layers=enc_layers,
        enc_out=enc_out,
        actor_layers=actor_layers,
        mean_net=dist_builder._mean_net,
        a_mean=state_dict[A_NORM_PREFIX + "_mean"],
        a_std=state_dict[A_NORM_PREFIX + "_std"],
        frame_dim=frame_dim,
        clip_action=clip_action)
    policy.to(device).eval()
    return policy


def build_reference_enc_policy(state_dict, actor_net, enc_net, device="cpu",
                               clip_action=True, obs_clip=OBS_NORM_CLIP):
    """Reference side of the encoder-export equivalence test, assembled from
    MimicKit's own classes and following MCWAMPEncModel.eval_actor +
    ppo_agent._decide_action + char_env._apply_action step for step."""
    obs_norm, a_norm = build_normalizers(state_dict, device=device, obs_clip=obs_clip)
    enc_layers, enc_out, actor_layers, dist_builder = build_enc_actor(
        state_dict, actor_net, enc_net, device=device)
    frame_dim, _, _ = get_enc_sizes(state_dict)

    bound_low = a_norm.get_mean() - a_norm.get_std()
    bound_high = a_norm.get_mean() + a_norm.get_std()

    def policy(obs):
        with torch.no_grad():
            norm_obs = obs_norm.normalize(obs)
            frame = norm_obs[..., :frame_dim]
            hist = norm_obs[..., frame_dim:]
            z = enc_out(enc_layers(hist))
            a_dist = dist_builder(actor_layers(torch.cat([frame, z], dim=-1)))
            a = a_norm.unnormalize(a_dist.mode)
            if (clip_action):
                a = torch.minimum(torch.maximum(a, bound_low), bound_high)
        return a

    return policy


def export_jit(policy, out_file):
    scripted = torch.jit.script(policy)
    scripted.save(out_file)
    return scripted


def export_onnx(policy, out_file, obs_size):
    """Optional. booster_deploy consumes no ONNX - it only ever calls
    torch.jit.load - so this exists for other tooling, not for deployment."""
    dummy = torch.zeros([1, obs_size], dtype=torch.float32)
    torch.onnx.export(policy, (dummy,), out_file,
                      input_names=["obs"], output_names=["dof_targets"],
                      dynamic_axes={"obs": {0: "batch"}, "dof_targets": {0: "batch"}})
    return


def max_abs_diff(a, b):
    return float(torch.max(torch.abs(a - b)).item())
