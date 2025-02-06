#!/bin/bash

set -eo pipefail

echo $UV_LINK_MODE
echo $PATH

sudo env "PATH=$PATH" uv pip install --no-build-isolation --editable \
  ./3rdparty/* \
  ./sub-packages/bionemo-* \
  -r requirements-cve.txt \
  -r requirements-test.txt \
  -r requirements-dev.txt

rm -rf ./sub-packages/bionemo-noodles/target
