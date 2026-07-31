"""Transplant MCWAMP steering weights into the soccer model layout.

The steering policy (obs = 237 char dims + 5 task dims = 242) and the soccer
policy (obs = 237 char dims + 5 steering-command dims + 7 soccer dims = 249)
share the [char obs | steering task obs] prefix: the soccer env fills the
steering slots with an auto command toward the ball, so the pretrained
velocity-tracking columns keep their meaning. The action space and the
discriminator (same dataset / key bodies) also match. Only the first layer of
the actor/critics and the obs normalizer depend on the obs layout, so we copy
the shared prefix columns from the steering checkpoint and keep the
soccer-initialized values for the new task columns.

Usage (inside the mimickit container, cwd = /workspace/MimicKit):
    python3.8 tools/transplant_steering_to_soccer.py \
        --src output/mcwamp_g1_steering_300m_seed1/int_models/model_0000004577.pt \
        --dst output/soccer_nan_sanity/model.pt \
        --out output/soccer_warmstart/model_init.pt
"""

import argparse
import os

import torch

# char obs (237) + steering task obs (5); the new soccer dims are appended
# after this prefix
PREFIX_OBS_DIM = 242


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="steering checkpoint (donor)")
    parser.add_argument("--dst", required=True, help="soccer checkpoint (layout template)")
    parser.add_argument("--out", required=True, help="output warm-start checkpoint")
    args = parser.parse_args()

    src = torch.load(args.src, map_location="cpu")
    dst = torch.load(args.dst, map_location="cpu")

    assert set(src.keys()) == set(dst.keys()), "checkpoint key sets differ"

    n_copied = n_prefix = 0
    for k in dst.keys():
        s, d = src[k], dst[k]
        if (not torch.is_tensor(s)):
            dst[k] = s
            n_copied += 1
        elif (s.shape == d.shape):
            dst[k] = s
            n_copied += 1
        else:
            # obs-dependent tensors: last dim is the obs dim
            assert s.shape[:-1] == d.shape[:-1], \
                "unexpected mismatch {}: {} vs {}".format(k, s.shape, d.shape)
            assert s.shape[-1] >= PREFIX_OBS_DIM and d.shape[-1] >= PREFIX_OBS_DIM, \
                "obs dim smaller than shared prefix for {}".format(k)
            d[..., :PREFIX_OBS_DIM] = s[..., :PREFIX_OBS_DIM]
            print("prefix-copied {}: {} -> {} (first {} dims)".format(
                k, tuple(s.shape), tuple(d.shape), PREFIX_OBS_DIM))
            n_prefix += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dst, args.out)
    print("copied {} tensors fully, {} by char-obs prefix".format(n_copied, n_prefix))
    print("wrote {}".format(args.out))
    return


if __name__ == "__main__":
    main()
