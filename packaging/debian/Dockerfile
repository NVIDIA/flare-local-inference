# syntax=docker/dockerfile:1
FROM rust:1.94.1-slim-bookworm AS toolchain

FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake dpkg-dev libibverbs-dev nasm \
    pkg-config protobuf-compiler python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=toolchain /usr/local/cargo /usr/local/cargo
COPY --from=toolchain /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH
WORKDIR /src
COPY flare/ flare/
COPY helios/ helios/
COPY eos/ eos/
COPY selene/selene/ selene/selene/
COPY ignition/recipes/qwen38/assets/ ignition/recipes/qwen38/assets/
COPY ignition/recipes/step5/assets/ ignition/recipes/step5/assets/
COPY ignition/recipes/kimi3/assets/ ignition/recipes/kimi3/assets/
COPY LICENSE NOTICE ./
COPY packaging/debian/ packaging/debian/
ARG PACKAGE_VERSION
ARG SOURCE_COMMIT
ARG SOURCE_DATE_EPOCH
ARG PACKAGE_MAINTAINER
ARG TARGETARCH
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,id=flare-deb-noble-${TARGETARCH},target=/src/flare/target \
    cargo build --locked --release --jobs 8 --manifest-path flare/Cargo.toml --bin flare \
    && packaging/debian/package.sh

FROM scratch AS artifact
COPY --from=build /out/ /
