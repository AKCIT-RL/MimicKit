#!/bin/bash
#SBATCH --job-name=mimickit_g1
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:1
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_gabrielruotolo/MimicKit/slurm_logs/%x_%j.log

# Submit with: sbatch slurm/train_dgx.sbatch args/<some_args_file>.txt
# Defaults to the domain-rand + stand + face-dir run if no arg_file is given.
#
# --signal=B:SIGUSR1@300 asks Slurm to send SIGUSR1 300s before the job's time limit /
# preemption -- mimickit/learning/base_agent.py::train_model catches it and forces an
# out-of-cycle checkpoint save (on top of the periodic one every iters_per_output
# iterations) so a maintenance-triggered kill doesn't lose more than a few seconds of
# progress. See that file for the handler.

set -euo pipefail

RAID_HOME=/raid/user_gabrielruotolo
REPO=$RAID_HOME/MimicKit
# Remaps $HOME (and /root, since we ask for :/root below) inside the container to a real
# /raid-backed directory. Needed because Apptainer runs as the submitting user (not root)
# by default, so anything writing to what it thinks is /root/.cache/... would otherwise
# land in the real home directory -- against cluster policy (never store in home).
# NOT YET VALIDATED against a real job -- first thing to check if the container fails to
# write caches/complains about permissions on the first run.
APPTAINER_HOME=$RAID_HOME/apptainer_home
SIF=$RAID_HOME/containers/mimickit.sif
ARG_FILE=${1:-args/amp_steering_g1_domainrand_stand_facedir_args.txt}

mkdir -p "$APPTAINER_HOME" "$REPO/output" "$REPO/slurm_logs"

apptainer exec --nv \
  --home "$APPTAINER_HOME:/root" \
  -B "$REPO/mimickit:/workspace/MimicKit/mimickit" \
  -B "$REPO/args:/workspace/MimicKit/args" \
  -B "$REPO/data/agents:/workspace/MimicKit/data/agents" \
  -B "$REPO/data/datasets:/workspace/MimicKit/data/datasets" \
  -B "$REPO/data/engines:/workspace/MimicKit/data/engines" \
  -B "$REPO/data/envs:/workspace/MimicKit/data/envs" \
  -B "$REPO/tools:/workspace/MimicKit/tools" \
  -B "$REPO/docs:/workspace/MimicKit/docs" \
  -B "$REPO/output:/workspace/MimicKit/output" \
  "$SIF" \
  python mimickit/run.py --mode train --arg_file "$ARG_FILE" --visualize false --video true --logger wandb
