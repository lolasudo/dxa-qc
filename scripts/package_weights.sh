#!/usr/bin/env bash
# Упаковка весов для передачи (архив + контрольная сумма); с --images - ещё и образы Docker.
#   scripts/package_weights.sh [<папка назначения, по умолчанию dist/>] [--images]
# На целевой машине: sha256sum -c *.sha256 && tar -xzf dxa-qc-weights-*.tar.gz -C backend/
source "$(dirname "$0")/common.sh"

dest="$ROOT/dist"
images=0
for arg in "$@"; do
    case "$arg" in
        --images) images=1 ;;
        -h | --help) sed -n '2,4p' "$0"; exit 0 ;;
        -*) die "неизвестный аргумент: $arg" ;;
        *) dest="$arg" ;;
    esac
done
require_weights
mkdir -p "$dest"
dest="$(abspath "$dest")"
stamp="$(date +%Y%m%d)"
archive="dxa-qc-weights-$stamp.tar.gz"

# Только то, что нужно сервису: без временных файлов и кэшей.
tar -czf "$dest/$archive" -C "$(dirname "$WEIGHTS_DIR")" --exclude='*.tmp*' "$(basename "$WEIGHTS_DIR")"
(cd "$dest" && sha256sum "$archive" >"$archive.sha256")
log "веса: $dest/$archive ($(du -h "$dest/$archive" | cut -f1))"

if [[ "$images" == 1 ]]; then
    require_docker
    img="dxa-qc-images-$stamp.tar.gz"
    images_list=(dxa-qc-backend:latest dxa-qc-frontend:latest)
    if docker image inspect dxa-qc-backend:gpu >/dev/null 2>&1; then images_list+=(dxa-qc-backend:gpu); fi
    docker save "${images_list[@]}" | gzip >"$dest/$img"
    (cd "$dest" && sha256sum "$img" >"$img.sha256")
    log "образы ${images_list[*]}: $dest/$img (на целевой машине: docker load -i $img)"
fi
