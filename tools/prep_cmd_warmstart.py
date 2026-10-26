"""Warm-start surgery for the velocity-command mode (doc 26, k94).

The kick checkpoints were trained with the steering slots annealed to ~0, so
their normalizer std there is ~1e-3 and the first-layer weights on those
columns are large. A nav command written into the slots would blow up the
first layer. This tool, on a copy of model.pt:
  - sets mean 0 / std 1 on the 5 steering slots of every actor frame
    (_obs_norm, current frame + history) and of the critic block
    (_critic_obs_norm, offset found from the slot statistics);
  - zeroes the first-layer weight columns that read those slots (encoder,
    actor, goal critic, aux critic).
With the slots ~0 (kick mode) the network output is unchanged up to the old
normalized value of the slots (|x| ~ 1e-2 times the old weight).

    python tools/prep_cmd_warmstart.py <in model.pt> <out model.pt> [--hist 30] [--frame 87] [--slot 75]
"""
import argparse
import torch

STEER_N = 5


def find_critic_block(mean, std, slot_std):
    """Offset of the steering slots in the critic obs: the 5 slots repeat the
    actor's (x, y, speed, x, y) statistics, slots 0/3 and 1/4 identical."""
    hits = []
    for o in range(0, mean.shape[0] - STEER_N + 1):
        s = std[o:o + STEER_N]
        if (torch.allclose(s, slot_std, rtol=0.05, atol=1e-5)):
            hits.append(o)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--hist", type=int, default=30)
    ap.add_argument("--frame", type=int, default=87)
    ap.add_argument("--slot", type=int, default=75)
    a = ap.parse_args()
    sd = torch.load(a.src, map_location="cpu")
    F, H, P = a.frame, a.hist, a.slot

    mean, std = sd["_obs_norm._mean"], sd["_obs_norm._std"]
    assert mean.shape[0] == F * (H + 1), mean.shape
    cols = torch.cat([torch.arange(f * F + P, f * F + P + STEER_N) for f in range(H + 1)])
    slot_std = std[P:P + STEER_N].clone()
    print("actor slot std before", slot_std.tolist())
    assert float(slot_std.max()) < 0.1, "steering slots are not near-constant: wrong checkpoint/offset?"

    c_mean, c_std = sd["_critic_obs_norm._mean"], sd["_critic_obs_norm._std"]
    hits = find_critic_block(c_mean, c_std, slot_std)
    assert len(hits) == 1, "critic steering block not unique: {}".format(hits)
    c0 = hits[0]
    print("critic steering slots at", c0, "std", c_std[c0:c0 + STEER_N].tolist())

    mean[cols] = 0.0
    std[cols] = 1.0
    c_mean[c0:c0 + STEER_N] = 0.0
    c_std[c0:c0 + STEER_N] = 1.0

    # encoder reads the history part of the obs (frames 1..H), actor the current frame
    enc_w = sd["_model._enc_layers.0.weight"]
    assert enc_w.shape[1] == F * H, enc_w.shape
    enc_cols = torch.cat([torch.arange(f * F + P, f * F + P + STEER_N) for f in range(H)])
    act_w = sd["_model._actor_layers.0.weight"]
    assert act_w.shape[1] > F, act_w.shape
    act_cols = torch.arange(P, P + STEER_N)
    for name, w, cc in (("enc", enc_w, enc_cols), ("actor", act_w, act_cols)):
        print(name, tuple(w.shape), "col norm before", w[:, cc].norm().item())
        w[:, cc] = 0.0
    for key in ("_model._critic_layers.0.weight", "_model._aux_critic_layers.0.weight"):
        w = sd[key]
        assert w.shape[1] == c_mean.shape[0], (key, w.shape)
        print(key, tuple(w.shape), "col norm before", w[:, c0:c0 + STEER_N].norm().item())
        w[:, c0:c0 + STEER_N] = 0.0
    torch.save(sd, a.dst)
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
