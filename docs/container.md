# Serving container

The root `Dockerfile` builds the Rust Flare executable with Helios and copies it
into a Debian slim image. Neither stage requires the CUDA toolkit or Python.
The runtime contains no Ignition, Heliopause, model weights, or compiler.
Helios loads the host's `libcuda.so.1` and replays kernels and graphs from a
prepared runtime package.

The host needs a compatible NVIDIA driver, Docker, and
[NVIDIA Container Toolkit configured for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
The package must match the GPU architecture and selected tensor-parallel size.
Prepare it outside the container; see [runtime packages](model-preparation.md#runtime-packages).

## Build

From the repository root:

```sh
docker build -t flare:local .
docker run --rm flare:local --version
docker run --rm flare:local run --help
```

The build context includes the Rust serving sources, Protobuf schema, and embedded
chat templates.
Cargo uses `flare/Cargo.lock`. The build installs a C/C++ toolchain, CMake,
NASM, and `protoc`; these remain in the build stage. BuildKit caches Cargo
downloads and compilation output across builds.

Linux AMD64 and ARM64 use the same Dockerfile. Build on the target architecture
or select it with Docker Buildx. ARM64 also includes libibverbs and its providers
because Flare enables the RoCE transport on that architecture. Multi-host RoCE
deployment additionally requires the host's RDMA devices and network setup.

## Run

This example exposes host GPU 0 and mounts a directory containing a prepared
package, its public publisher profile, and an already approved trust store:

```sh
docker run --rm --name flare \
  --gpus 'device=0,capabilities=compute' \
  --ulimit memlock=-1:-1 \
  --stop-timeout 60 \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src=/absolute/path/to/packages,dst=/models,readonly \
  flare:local run /models/model.safetensors \
  --publisher-keys /models/publisher.json \
  --trust-store /models/trust.toml \
  --gpus 0 --host 0.0.0.0 --port 8000
```

The image runs as UID/GID `10001:10001`; mounted files and parent directories
must be readable by that user. Alternatively, pass Docker
`--user "$(id -u):$(id -g)"` and an explicit `--trust-store` path to use the host
file owner's identity. Mount only runtime files, including public keys;
private signing keys are preparation inputs. For first-use approval, add `-it`
and mount a writable trust directory separately. Pass its `trust.toml` path with
`--trust-store` so acceptance survives container removal. Unattended approval
uses the exact fingerprint and namespace described in
[signing and local development](cli-design.md#signing-and-local-development).

Docker's `--gpus` selects host devices. Flare's `--gpus` selects devices from the
container's visible list. For a two-rank package on host GPUs 4 and 5, use
`--gpus '"device=4,5",capabilities=compute'` on Docker and `--gpus 0,1` on Flare.
The NVIDIA runtime's `compute` capability mounts the driver libraries;
Flare does not require CUDA runtime, cuBLAS, or NCCL libraries in the image.
See [NVIDIA driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities).

Bind Flare to `0.0.0.0` inside the container so Docker's published port reaches
it. Add `--api-key-file` with a mounted secret when authentication is required.
The executable is PID 1 and handles SIGTERM directly; `docker stop` allows
native replay to drain within the configured stop timeout.

## Downloading packages

Mount a writable directory as the Hub cache to reuse downloads across containers.
The directory must be owned by UID/GID `10001:10001`, or run with its owner's
identity through Docker's `--user` flag. For example:

```sh
mkdir -p "$HOME/.cache/huggingface/hub"
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$HOME/.cache/huggingface/hub,dst=/cache" \
  flare:local download ORG/MODEL \
  --revision gb202 --cache-dir /cache
```

`ORG/MODEL` is a repository placeholder. An explicit revision permits downloading
without exposing GPUs. For automatic selection, expose the intended host GPUs
with Docker's `--gpus` and select their container-local indices with Flare's
`--gpus`. `flare run ORG/NAME` accepts the same download options, then loads the
cached package. Mount a persistent trust directory and pass `--trust-store` when
running with a host UID absent from the image's user database. Private repositories
add `--token-file` pointing to a separately mounted token file. See
[published packages](cli-design.md#downloading-published-packages) for revision,
filename selection, and offline behavior.

Disk-backed embedding tables and disk prefix-cache offload use Linux `io_uring`.
The container's seccomp policy must permit `io_uring_setup`, `io_uring_enter`,
and `io_uring_register` for those modes. Docker's default policy can block them;
use a policy allowing those calls, such as
`selene-manager/deploy/seccomp-io-uring.json` (Docker's default plus io_uring). Prefix-cache disk directories also need a
writable bind mount owned by the serving user. `--ulimit memlock=-1:-1` permits
the pinned host storage used for transfers.
