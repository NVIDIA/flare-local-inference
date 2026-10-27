# Build targets for Flare. These mirror the documented setup in
# docs/model-preparation.md ("Environment setup"); that guide stays authoritative
# for dependency policy, FlashInfer pins, and GPU/toolchain requirements.

UV ?= uv
CARGO ?= cargo

.PHONY: all deps sync build extension clean

all: sync build extension

# System build dependencies (Ubuntu). Explicit target because it needs sudo.
deps:
	sudo apt-get install -y build-essential cmake ninja-build pkg-config \
		protobuf-compiler libssl-dev python3-dev liburing-dev

# Preparation environment from uv.lock, including the prepare and test extras.
sync:
	$(UV) sync --locked --extra prepare --extra test

# Native CLI: flare/target/release/flare
build:
	$(CARGO) build --release --manifest-path flare/Cargo.toml

# Rebuild the installed flare_heliopause extension after changing
# heliopause/ or Helios code it uses.
extension:
	$(UV) pip install --python .venv/bin/python --no-deps ./heliopause

clean:
	$(CARGO) clean --manifest-path flare/Cargo.toml
