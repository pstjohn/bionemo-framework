# syntax=docker/dockerfile:1.4
# FROM nvcr.io/nvidia/pytorch:26.01-py3

# Note: there's currently a bug in GQA + CP on Hopper but it's fixed in cuDNN 9.18. This image is NVIDIA-internal, but
# leaving this as a note in to remove once the bug is fixed in an nvidia/pytorch image.
FROM gitlab-master.nvidia.com/dl/transformerengine/transformerengine:main-pytorch-py3-base

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

WORKDIR /workspace/bionemo
COPY . .
