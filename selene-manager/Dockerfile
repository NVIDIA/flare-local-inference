# Build from the Flare repository root:
#   docker build -f selene-manager/Dockerfile -t selene .
FROM rust:1.94.1-bookworm AS build
# prost-build needs protoc; the rust image already carries the C toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY eos/Cargo.toml ./eos/
COPY eos/src ./eos/src
COPY selene/Cargo.toml selene/build.rs ./selene/
COPY selene/src ./selene/src
COPY selene/examples ./selene/examples
COPY selene/proto ./selene/proto
COPY selene-manager/Cargo.toml selene-manager/Cargo.lock ./selene-manager/
COPY selene-manager/src ./selene-manager/src
COPY selene-manager/tests ./selene-manager/tests
WORKDIR /build/selene-manager
RUN cargo build --release --locked

FROM debian:bookworm-slim
COPY --from=build /build/selene-manager/target/release/selene /usr/local/bin/selene
ENTRYPOINT ["/usr/local/bin/selene"]
