#!/bin/bash

set -eo pipefail

uv pip install --no-build-isolation --editable \
  ./3rdparty/* \
  ./sub-packages/bionemo-*

rm -rf /tmp/*
rm -rf ./sub-packages/bionemo-noodles/target
