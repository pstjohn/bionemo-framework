# BioNeMo Framework

some change
some other change

[![Click here to deploy.](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg)](https://console.brev.dev/launchable/deploy/now?launchableID=env-2pPDA4sJyTuFf3KsCv5KWRbuVlU)
[![Docs Build](https://img.shields.io/github/actions/workflow/status/NVIDIA/bionemo-framework/pages/pages-build-deployment?label=docs-build)](https://nvidia.github.io/bionemo-framework)
[![Test Status](https://github.com/NVIDIA/bionemo-framework/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/NVIDIA/bionemo-framework/actions/workflows/unit-tests.yml)
[![Latest Tag](https://img.shields.io/github/v/tag/NVIDIA/bionemo-framework?label=latest-version)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework/tags)
[![codecov](https://codecov.io/gh/NVIDIA/bionemo-framework/branch/main/graph/badge.svg?token=XqhegdZRqB)](https://codecov.io/gh/NVIDIA/bionemo-framework)

NVIDIA BioNeMo Framework is a is a comprehensive suite of programming tools, libraries, and models designed for computational drug discovery.
It accelerates the most time-consuming and costly stages of building and adapting biomolecular AI models by providing
domain-specific, optimized models and tooling that are easily integrated into GPU-based computational resources for the
fastest performance on the market. You can access BioNeMo Framework as a free community resource here in this repository
or learn more at <https://www.nvidia.com/en-us/clara/bionemo/> about getting an enterprise license for improved
expert-level support.

## Structure of the Framework

The `bionemo-framework` is organized into independently installable namespace packages. These are located under the
`sub-packages/` directory. Please refer to [PEP 420 – Implicit Namespace Packages](https://peps.python.org/pep-0420/)
for details.


## Documentation Resources

- **Official Documentation:** For user guides, API references, and troubleshooting, visit our [official documentation](https://docs.nvidia.com/bionemo-framework/latest/).
- **In-Progress Documentation:** To explore the latest features and developments, check the documentation reflecting the current state of the `main` branch [here](https://nvidia.github.io/bionemo-framework/). Note that this may include references to features or APIs that are not yet finalized.

## Getting Started with BioNeMo Framework

Full documentation on using the BioNeMo Framework is provided in our documentation:
<https://docs.nvidia.com/bionemo-framework/latest/user-guide/>. To simplify the integration of optimized third-party dependencies, BioNeMo is primarily distributed as a containerized library. You can download the latest released container for the BioNeMo Framework from
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework). To launch a pre-built container, you can use the brev.dev launchable [![ Click here to deploy.](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg)](https://console.brev.dev/launchable/deploy/now?launchableID=env-2pPDA4sJyTuFf3KsCv5KWRbuVlU) or execute the following command:

```bash
docker run --rm -it \
  --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/clara/bionemo-framework:nightly \
  /bin/bash
```

### Setting up a local development environment

#### Initializing 3rd-party dependencies as git submodules

The NeMo and Megatron-LM dependencies are included as git submodules in bionemo2. The pinned commits for these submodules represent the "last-known-good" versions of these packages
that are confirmed to be working with bionemo2 (and those that are tested in CI).

To initialize these sub-modules when cloning the repo, add the `--recursive` flag to the git clone command:

```bash
git clone --recursive git@github.com:NVIDIA/bionemo-framework.git
cd bionemo-framework
```

To download the pinned versions of these submodules within an existing git repository, run

```bash
git submodule update --init --recursive
```

Different branches of the repo can have different pinned versions of these third-party submodules. Ensure submodules are automatically updated after switching branches or pulling updates by configuring git with:


```bash
git config submodule.recurse true
```

**NOTE**: this setting will not download **new** or remove **old** submodules with the branch's changes.
You will have to run the full `git submodule update --init --recursive` command in these situations.

#### Build the Docker Image Locally


With a locally cloned repository and initialized submodules, build the BioNeMo container using:

```bash
docker buildx build . -t my-container-tag
```


#### VSCode Devcontainer for Interactive Debugging

We distribute a [development container](https://devcontainers.github.io/) configuration for vscode
(`.devcontainer/devcontainer.json`) that simplifies the process of local testing and development. Opening the
bionemo-framework folder with VSCode should prompt you to re-open the folder inside the devcontainer environment.

> [!NOTE]
> The first time you launch the devcontainer, it may take a long time to build the image. Building the image locally
> (using the command shown above) will ensure that most of the layers are present in the local docker cache.

### Quick Start

See the [tutorials pages](https://docs.nvidia.com/bionemo-framework/latest/user-guide/examples/bionemo-esm2/pretrain/)
for example applications and getting started guides.
