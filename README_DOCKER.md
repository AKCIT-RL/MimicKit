# MimicKit — Docker Guide
## Host Prerequisites

### 1. Docker and Docker Compose

- **Docker Engine** 26.0.0 or newer
- **Docker Compose** 2.25.0 or newer

Install using the [official Docker documentation](https://docs.docker.com/engine/install/).

### 2. NVIDIA Container Toolkit

Required for GPU execution. Follow the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### 3. NGC Account and Login

The base image `nvcr.io/nvidia/isaac-lab:2.1.0` requires NGC authentication:

1. Create an account on the [NVIDIA Developer Program](https://developer.nvidia.com/)
2. Generate an NGC API key in [NGC Setup](https://ngc.nvidia.com/setup/api-key)
3. Log in to Docker:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: your NGC API key
```

### 4. System Requirements

- **OS:** Ubuntu 22.04 (Linux x64)
- **RAM:** 32 GB or more
- **GPU VRAM:** 16 GB or more
- **NVIDIA driver:** 575.x or newer (this Dockerfile pins torch/torchvision to the cu128
  build to match; if the target machine has a newer driver that supports cu130, the base
  image's stock PyTorch may work without the override in the Dockerfile)

### 5. Repository Location

Keep the repository under `/home` on the host (avoids snap/Docker path issues on some setups).

## Building the Image

```bash
cd MimicKit
docker compose build
```

The build pulls the NGC base image (~12 GB) and bakes in the G1 assets/motions
(`data/assets/g1`, `data/motions/g1`). First build may take several minutes.

## Running the Container

### Interactive mode (bash)

```bash
docker compose run --rm mimickit
```

Opens a shell inside the container. Working directory is `/workspace/MimicKit`.

### Run a command directly (no shell)

**Smoke test:**
```bash
docker compose run --rm mimickit \
  python mimickit/run.py --mode test --arg_file args/amp_steering_g1_domainrand_stand_facedir_args.txt \
  --visualize false --video true
```

**Train (with wandb logging):**
```bash
docker compose run --rm mimickit \
  python mimickit/run.py --mode train --arg_file args/amp_steering_g1_domainrand_stand_facedir_args.txt \
  --visualize false --video true --logger wandb
```

**Test a specific checkpoint:**
```bash
docker compose run --rm mimickit \
  python mimickit/run.py --mode test --arg_file args/amp_steering_g1_domainrand_stand_args.txt \
  --model_file output/g1_locomotion_domainrand_stand/model.pt --num_envs 4 \
  --visualize false --video true
```

## Weights & Biases

The wandb entity is hardcoded in `mimickit/util/wandb_logger.py` (not an env var), so it
comes along automatically since `mimickit/` is bind-mounted. You only need to authenticate:

```bash
docker compose run --rm -e WANDB_API_KEY=your-key mimickit \
  python mimickit/run.py --mode train --arg_file args/<...>.txt --logger wandb
```

Or set `WANDB_API_KEY` in your host shell before `docker compose run` — it's already wired
through in `docker-compose.yaml`.

## What's Baked In vs Bind-Mounted

| Baked into the image (rebuild to update) | Bind-mounted from the host repo (edits apply live) |
|---|---|
| Isaac Lab, torch/torchvision, `requirements.txt` deps | `mimickit/` (all source code) |
| `data/assets/g1/`, `data/motions/g1/` | `args/`, `data/agents/`, `data/datasets/`, `data/engines/`, `data/envs/` |
| | `tools/`, `docs/` |
| | `output/` (checkpoints/logs persist back to the host) |

If you add a new motion clip or asset file under `data/assets/g1`/`data/motions/g1`, you need
to `docker compose build` again — those paths aren't bind-mounted on purpose (see
`docker-compose.yaml` comments).

## Volumes and Cache

Named volumes mount Isaac Sim caches to avoid recompiling shaders/kit resources on every run:
`isaac-cache-kit`, `isaac-cache-ov`, `isaac-cache-pip`, `isaac-cache-gl`, `isaac-cache-compute`,
`isaac-logs`, `isaac-data`. These are Docker-managed volumes, not host directories.

## Deploying to a New Machine (e.g. an OVX)

1. Copy/clone the MimicKit repo to the new machine (needed for the bind-mounted paths above —
   `data/assets/g1`/`data/motions/g1` aren't required on the host copy since they're baked
   into the image, but a full clone is simplest).
2. `docker login nvcr.io` on the new machine.
3. `docker compose build`.
4. Run the same smoke test command above to confirm the image works before a real training run.

## EULA

By running this container you agree to the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement).

## Troubleshooting

- **GPU errors:** Ensure the NVIDIA Container Toolkit is installed and `nvidia-smi` works on
  the host.
- **NGC login errors:** Confirm `docker login nvcr.io` completed successfully.
- **Torch/CUDA mismatch:** If the build fails installing `torch==2.11.0+cu128`, check the
  target driver's max supported CUDA version (`nvidia-smi` header) and adjust the pinned
  torch/torchvision versions and `--index-url` in the `Dockerfile` accordingly.
- **Xvfb/video capture fails:** `apt-get install -y xvfb ffmpeg` runs at build time; if a
  headless render still fails, check `mimickit/util/display.py`'s error output for the
  specific `Xvfb` invocation failure.
