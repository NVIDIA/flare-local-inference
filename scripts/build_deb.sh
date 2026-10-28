#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_deb.sh --maintainer 'Name <email>' [--version VERSION] [--output DIR]

Build a binary-only Flare Debian package in Ubuntu 24.04 on the native Docker
architecture. Requires Docker Buildx and Git. Defaults to dist/debian and a
version derived from Cargo.toml, the commit timestamp, and the source commit.
EOF
}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output="$root/dist/debian"
maintainer=
version=
while (($#)); do
    case "$1" in
        --maintainer|--version|--output)
            if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                usage >&2; exit 2
            fi
            case "$1" in
                --maintainer) maintainer=$2 ;;
                --version) version=$2 ;;
                --output) output=$2 ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ ! "$maintainer" =~ ^[^\<\>]+\ \<[^\<\>\ ]+@[^\<\>\ ]+\>$ || "$maintainer" == *$'\n'* ]]; then
    printf 'Supply --maintainer with a name and contact email.\n' >&2
    exit 2
fi
commit=$(git -C "$root" rev-parse HEAD)
epoch=$(git -C "$root" show -s --format=%ct HEAD)
if [[ -z "$version" ]]; then
    base=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/flare/Cargo.toml" | head -1)
    stamp=$(date -u -d "@$epoch" +%Y%m%d%H%M%S)
    version="${base}~git${stamp}.${commit:0:8}-1"
fi
if [[ ! "$version" =~ ^[0-9][A-Za-z0-9.+:~\-]*$ ]]; then
    printf 'Invalid Debian package version.\n' >&2
    exit 2
fi
if ! git -C "$root" diff --quiet HEAD -- flare helios eos selene selene-manager; then
    printf 'Commit native source changes before building a release package.\n' >&2
    exit 1
fi
mkdir -p "$output"
docker buildx build --file "$root/packaging/debian/Dockerfile" \
    --build-arg "PACKAGE_VERSION=$version" \
    --build-arg "PACKAGE_MAINTAINER=$maintainer" \
    --build-arg "SOURCE_COMMIT=$commit" \
    --build-arg "SOURCE_DATE_EPOCH=$epoch" \
    --output "type=local,dest=$output" --progress plain "$root"
