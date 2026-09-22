"""Expand a v1 soccer encoder checkpoint (87-dim measurable frame) to the
Fase 1 layout with K robot slots appended to every frame's task block.

Per-frame column expansion (the history is a tiling of frames, so every
frame gets ROBOT_SLOT_DIM * K new columns at its tail):
  * _obs_norm._mean/_std        : (1+H)*87 -> (1+H)*(87+4K); new mean 0 / std 1
  * _model._enc_layers.0.weight : columns over H frames, new columns zero
  * _model._actor_layers.0.weight: [frame | latent] -> [frame + 4K | latent],
                                   new frame columns zero (latent columns keep
                                   their position relative to the frame end)
  * _critic_obs_norm / critic + aux critic layers.0: the critic obs is
                                   [char | privileged task block | ball vel 2 |
                                   (5M obstacle priv)], so 4K zero columns go
                                   BEFORE the last CRITIC_TAIL_DIM columns and
                                   5M zero columns at the end (mean 0 / std 1)
  * _model._dec_out.weight/bias : 4 -> 4 + 4K rows (new rows zero)

Zero weights on the new inputs make the expanded policy produce exactly the
same actions as the source on any obs whose new dims are ignored, so the
warm start is behaviourally the source policy (verified by the tool).

Usage (inside the mimickit container, cwd = /workspace/MimicKit):
    python3.8 tools/expand_soccer_obs_checkpoint.py \
        --src output/t1_soccer_w2_enc_c11_seed1/model.pt \
        --out output/t1_soccer_w2_enc_c11_seed1/model_o1_init.pt \
        --frame_dim 87 --hist_steps 30 --num_robot_slots 2 --num_obstacles 2
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mimickit"))

import envs.soccer_util as soccer_util  # noqa: E402

# TaskSoccerEnv._compute_critic_obs (measurable mode): dims after the
# privileged task block in the v1 layout = true planar ball velocity
CRITIC_TAIL_DIM = 2


def _expand_critic_columns(w, extra_slots, extra_priv):
    """[..., prefix | tail] -> [..., prefix | 0*extra_slots | tail | 0*extra_priv]."""
    lead = w.shape[:-1]
    prefix = w[..., :w.shape[-1] - CRITIC_TAIL_DIM]
    tail = w[..., w.shape[-1] - CRITIC_TAIL_DIM:]
    return torch.cat([prefix, torch.zeros(*lead, extra_slots, dtype=w.dtype), tail,
                      torch.zeros(*lead, extra_priv, dtype=w.dtype)], dim=-1)


def _expand_critic_norm(v, extra_slots, extra_priv, fill):
    prefix = v[: v.shape[0] - CRITIC_TAIL_DIM]
    tail = v[v.shape[0] - CRITIC_TAIL_DIM:]
    return torch.cat([prefix, torch.full([extra_slots], fill, dtype=v.dtype), tail,
                      torch.full([extra_priv], fill, dtype=v.dtype)])


def _expand_frame_columns(w, frame_dim, num_frames, extra, tail_cols=0):
    """Insert `extra` zero columns at the end of each of `num_frames` frames
    of width frame_dim along the last dim; `tail_cols` trailing columns
    (e.g. the latent) are kept after the frames."""
    assert w.shape[-1] == num_frames * frame_dim + tail_cols, \
        "expected {} = {} x {} + {} columns, got {}".format(
            num_frames * frame_dim + tail_cols, num_frames, frame_dim, tail_cols, w.shape[-1])
    frames = w[..., :num_frames * frame_dim].reshape(*w.shape[:-1], num_frames, frame_dim)
    pad = torch.zeros(*w.shape[:-1], num_frames, extra, dtype=w.dtype)
    frames = torch.cat([frames, pad], dim=-1).reshape(*w.shape[:-1], num_frames * (frame_dim + extra))
    if (tail_cols > 0):
        frames = torch.cat([frames, w[..., num_frames * frame_dim:]], dim=-1)
    return frames


def _expand_frame_norm(v, frame_dim, num_frames, extra, fill):
    assert v.shape == (num_frames * frame_dim,), (v.shape, num_frames, frame_dim)
    frames = v.reshape(num_frames, frame_dim)
    pad = torch.full([num_frames, extra], fill, dtype=v.dtype)
    return torch.cat([frames, pad], dim=-1).reshape(-1)


def expand_checkpoint(src, frame_dim, hist_steps, num_robot_slots, num_obstacles):
    extra = soccer_util.ROBOT_SLOT_DIM * num_robot_slots
    extra_critic = soccer_util.OBSTACLE_PRIV_DIM * num_obstacles
    extra_recon = 4 * num_robot_slots
    num_frames = 1 + hist_steps
    out = src.__class__(src)

    out["_obs_norm._mean"] = _expand_frame_norm(src["_obs_norm._mean"], frame_dim, num_frames, extra, 0.0)
    out["_obs_norm._std"] = _expand_frame_norm(src["_obs_norm._std"], frame_dim, num_frames, extra, 1.0)
    out["_model._enc_layers.0.weight"] = _expand_frame_columns(
        src["_model._enc_layers.0.weight"], frame_dim, hist_steps, extra)
    w = src["_model._actor_layers.0.weight"]
    latent_dim = w.shape[-1] - frame_dim
    assert latent_dim > 0, "actor input smaller than a frame"
    out["_model._actor_layers.0.weight"] = _expand_frame_columns(w, frame_dim, 1, extra,
                                                                 tail_cols=latent_dim)

    if (extra > 0 or extra_critic > 0):
        out["_critic_obs_norm._mean"] = _expand_critic_norm(src["_critic_obs_norm._mean"], extra, extra_critic, 0.0)
        out["_critic_obs_norm._std"] = _expand_critic_norm(src["_critic_obs_norm._std"], extra, extra_critic, 1.0)
        for name in ["_model._critic_layers.0.weight", "_model._aux_critic_layers.0.weight"]:
            out[name] = _expand_critic_columns(src[name], extra, extra_critic)

    if (extra_recon > 0):
        dw = src["_model._dec_out.weight"]
        db = src["_model._dec_out.bias"]
        out["_model._dec_out.weight"] = torch.cat([dw, torch.zeros(extra_recon, dw.shape[1], dtype=dw.dtype)], dim=0)
        out["_model._dec_out.bias"] = torch.cat([db, torch.zeros(extra_recon, dtype=db.dtype)])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame_dim", type=int, default=87)
    parser.add_argument("--hist_steps", type=int, default=30)
    parser.add_argument("--num_robot_slots", type=int, default=2)
    parser.add_argument("--num_obstacles", type=int, default=2)
    args = parser.parse_args()

    src = torch.load(args.src, map_location="cpu")
    out = expand_checkpoint(src, args.frame_dim, args.hist_steps,
                            args.num_robot_slots, args.num_obstacles)
    for k in out:
        if (torch.is_tensor(out[k]) and out[k].shape != src[k].shape):
            print("expanded {}: {} -> {}".format(k, tuple(src[k].shape), tuple(out[k].shape)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print("wrote {}".format(args.out))
    return


if __name__ == "__main__":
    main()
