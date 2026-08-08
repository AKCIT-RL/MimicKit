# Base: Isaac Lab 2.1.0 (Isaac Sim 4.5.0, Python 3.10) -- same base image the
# CopyCat/train/whole_body_tracking (BeyondMimic) Dockerfile uses, which the local
# env_isaaclab conda env is *already* built on (confirmed via `pip show isaaclab`:
# editable install pointing at .../whole_body_tracking/IsaacLab/source/isaaclab).
FROM nvcr.io/nvidia/isaac-lab:2.1.0

# xvfb: mimickit/util/display.py starts a virtual display for headless camera rendering
# (used by --video true). ffmpeg: moviepy (in requirements.txt) shells out to it to encode
# video. Neither ships in the base image or in requirements.txt -- both are system packages.
RUN apt-get update && apt-get install -y xvfb ffmpeg && rm -rf /var/lib/apt/lists/*

# The base image ships PyTorch built for cu130 (needs driver >=580). The training machine's
# driver is 575.x, which tops out at cu128 -- reinstall with a matching build. Versions
# confirmed live in env_isaaclab via `pip show torch torchvision`.
RUN /isaac-sim/python.sh -m pip install \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Same patched IsaacLab fork already used by env_isaaclab locally (confirmed via
# `pip show isaaclab` -> Editable project location pointing at this exact fork/branch).
RUN git clone --depth 1 -b whole_body_tracking \
        https://github.com/AKCIT-RL/IsaacLab.git /opt/IsaacLab && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab --no-deps && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab_assets --no-deps && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab_tasks --no-deps && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab_mimic --no-deps && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab_rl --no-deps

WORKDIR /workspace/MimicKit

COPY requirements.txt .
RUN /isaac-sim/python.sh -m pip install -r requirements.txt

# G1 assets/motions baked into the image (stable, rarely change -- unlike mimickit/, args/,
# data/envs, etc., which stay bind-mounted from the host repo in docker-compose.yaml so
# editing them doesn't require a rebuild). This keeps the image runnable standalone even
# before the repo has been copied to a new machine.
COPY data/assets/g1 ./data/assets/g1
COPY data/motions/g1 ./data/motions/g1

# Interactive shell by default (matches the whole_body_tracking Dockerfile convention).
ENTRYPOINT ["/bin/bash"]
