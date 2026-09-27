#!/usr/bin/env bash
# Сборка образов.
#   scripts/build.sh                  CPU-образы backend + frontend
#   scripts/build.sh --gpu            backend с CUDA-сборкой torch
#   scripts/build.sh --bake-weights   дополнительно dxa-qc-backend:baked - веса внутри образа
source "$(dirname "$0")/common.sh"

BAKE=0
for arg in "$@"; do
    case "$arg" in
        --gpu) enable_gpu ;;
        --bake-weights) BAKE=1 ;;
        -h | --help) sed -n '2,6p' "$0"; exit 0 ;;
        *) die "неизвестный аргумент: $arg" ;;
    esac
done

require_docker
log "сборка образов (GPU=${USE_GPU:-0})"
compose build

if [[ "$BAKE" == 1 ]]; then
    require_weights
    index="https://download.pytorch.org/whl/cpu"
    if [[ "${USE_GPU:-0}" == 1 ]]; then index="https://download.pytorch.org/whl/cu128"; fi
    log "сборка dxa-qc-backend:baked (веса из $WEIGHTS_DIR)"
    docker buildx build --load --target baked \
        --build-arg TORCH_INDEX_URL="$index" \
        --build-context weights="$WEIGHTS_DIR" \
        -t dxa-qc-backend:baked "$ROOT/backend"
fi
log "готово"
