#!/usr/bin/env bash
#
# Build the rembed development-environment Docker image from the Nix flake and,
# optionally, push it to a registry (e.g. Docker Hub).
#
# The image ships the full dev environment (Rust toolchain, R, Python, CGAL,
# native libs, Postgres client) — NOT compiled binaries. Researchers mount the
# source at /work and build/run the experiments inside the container:
#
#     docker run -it -v "$PWD:/work" <image> \
#         cargo run --bin embedder-cli -- ...
#
# A push publishes two tags pointing at the same image: the primary TAG (git
# short SHA by default, for traceability) and `latest`. Set PUSH_LATEST=0 to
# publish only the primary tag.
#
# Usage:
#     ./build_docker.sh                      # build + load into local docker
#     ./build_docker.sh --push               # build, push :<git-sha> and :latest
#     REGISTRY=docker.io/truedoctor ./build_docker.sh --push
#     TAG=v0.1.0 ./build_docker.sh --push    # push :v0.1.0 and :latest
#     PUSH_LATEST=0 ./build_docker.sh --push # push only the primary tag
#     ./build_docker.sh --push --skopeo      # push without a docker daemon
#
set -euo pipefail

# Directory this script lives in, so co-located files resolve no matter the CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration (override via environment) --------------------------------

# Registry namespace to push to. For Docker Hub this is docker.io/<username>.
REGISTRY="${REGISTRY:-docker.io/truedoctor}"
# Image repository name (matches `name` in the flake's dockerImage).
IMAGE_NAME="${IMAGE_NAME:-rembed-env}"
# Tag to publish. Defaults to the short git commit so images are traceable.
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
# Also move the `latest` tag to this build. Set PUSH_LATEST=0 to opt out.
PUSH_LATEST="${PUSH_LATEST:-1}"

FULL_REF="${REGISTRY}/${IMAGE_NAME}:${TAG}"
LATEST_REF="${REGISTRY}/${IMAGE_NAME}:latest"

# Nothing extra to do if the primary tag is already `latest`.
if [ "$TAG" = "latest" ]; then
    PUSH_LATEST=0
fi

PUSH=0
USE_SKOPEO=0
for arg in "$@"; do
    case "$arg" in
        --push)   PUSH=1 ;;
        --skopeo) USE_SKOPEO=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

# --- Build the image tarball with Nix ----------------------------------------

echo ">> Building image with Nix (this can take a while on first run)..."
IMAGE_TARBALL="$(nix build .#dockerImage --no-link --print-out-paths)"
echo ">> Built: ${IMAGE_TARBALL}"
echo ">> Size:  $(du -h "${IMAGE_TARBALL}" | cut -f1)"

if [ "$PUSH" -eq 0 ]; then
    # Local-only: load into the docker daemon so it can be run immediately.
    echo ">> Loading into local docker daemon..."
    docker load < "${IMAGE_TARBALL}"
    echo ">> Done. Run it with:"
    echo "     docker run -it -v \"\$PWD:/work\" ${IMAGE_NAME}:latest"
    exit 0
fi

# --- Push to the registry ----------------------------------------------------

if [ "$USE_SKOPEO" -eq 1 ]; then
    # Daemonless path: copy the tarball straight to the registry. Requires a
    # prior `skopeo login docker.io` (or ~/.docker/config.json credentials).
    #
    # skopeo needs a trust policy; systems without containers-common have none,
    # so we ship a permissive one alongside this script and point skopeo at it.
    POLICY="${SCRIPT_DIR}/containers-policy.json"
    echo ">> Pushing ${FULL_REF} via skopeo (no docker daemon needed)..."
    skopeo copy --policy "${POLICY}" \
        "docker-archive:${IMAGE_TARBALL}" "docker://${FULL_REF}"
    if [ "$PUSH_LATEST" -eq 1 ]; then
        # Same manifest under a second name; only the manifest is re-uploaded.
        echo ">> Also tagging ${LATEST_REF}..."
        skopeo copy --policy "${POLICY}" \
            "docker-archive:${IMAGE_TARBALL}" "docker://${LATEST_REF}"
    fi
else
    # Docker path: load, tag, push. Requires `docker login` first.
    echo ">> Loading into local docker daemon..."
    LOADED="$(docker load < "${IMAGE_TARBALL}" | sed -n 's/^Loaded image: //p')"
    echo ">> Loaded: ${LOADED}"
    echo ">> Tagging ${LOADED} -> ${FULL_REF}"
    docker tag "${LOADED}" "${FULL_REF}"
    echo ">> Pushing ${FULL_REF}..."
    docker push "${FULL_REF}"
    if [ "$PUSH_LATEST" -eq 1 ]; then
        # Same manifest under a second name; only the manifest is re-uploaded.
        echo ">> Tagging + pushing ${LATEST_REF}..."
        docker tag "${LOADED}" "${LATEST_REF}"
        docker push "${LATEST_REF}"
    fi
fi

echo ">> Published: ${FULL_REF}"
[ "$PUSH_LATEST" -eq 1 ] && echo ">> Published: ${LATEST_REF}"
echo ">> Researchers can now run:"
echo "     docker run -it --cap-add PERFMON -v \"\$PWD:/work\" ${LATEST_REF}"
