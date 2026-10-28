# syntax=docker/dockerfile:1

FROM rust:1.94.1-slim-trixie AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake nasm pkg-config protobuf-compiler \
    && if [ "$(dpkg --print-architecture)" = arm64 ]; then \
        apt-get install -y --no-install-recommends libibverbs-dev; \
    fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY flare/ flare/
COPY helios/ helios/
COPY eos/ eos/
COPY selene/ selene/
COPY selene-manager/ selene-manager/
COPY ignition/recipes/qwen38/assets/ ignition/recipes/qwen38/assets/
COPY ignition/recipes/step5/assets/ ignition/recipes/step5/assets/
COPY ignition/recipes/kimi3/assets/ ignition/recipes/kimi3/assets/

ARG TARGETARCH
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,id=flare-target-${TARGETARCH},target=/src/flare/target \
    cargo build --locked --release --manifest-path flare/Cargo.toml --bin flare \
    && install -Dm755 flare/target/release/flare /out/flare \
    && strip /out/flare

FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgcc-s1 \
    && if [ "$(dpkg --print-architecture)" = arm64 ]; then \
        apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers; \
    fi \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 flare \
    && useradd --uid 10001 --gid flare --create-home --no-log-init \
        --shell /usr/sbin/nologin flare

COPY --from=build /out/flare /usr/local/bin/flare
COPY LICENSE NOTICE /usr/share/doc/flare/

USER flare
WORKDIR /home/flare
EXPOSE 8000
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/flare"]
CMD ["--help"]
