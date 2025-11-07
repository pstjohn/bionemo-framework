# syntax=docker/dockerfile:1.4
FROM gitlab-master.nvidia.com:5005/dl/transformerengine/transformerengine:main-pytorch-py3-devel

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

WORKDIR /workspace/bionemo
COPY . .
