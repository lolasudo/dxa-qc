# Общие функции скриптов. Подключается через `source`, сам не запускается.
# shellcheck shell=bash
set -euo pipefail

# Путь каталога в форме, понятной docker: в Git Bash на Windows - C:/..., иначе обычный.
_pwd() { pwd -W 2>/dev/null || pwd; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && _pwd)"
WEIGHTS_DIR="${DXA_QC_WEIGHTS:-$ROOT/backend/weights}"
# shellcheck disable=SC2034  # используется скриптами, подключающими этот файл
BACKEND_IMAGE="${DXA_QC_BACKEND_IMAGE:-dxa-qc-backend:latest}"
REQUIRED_WEIGHTS=(region_classifier.npz quality_spine.npz quality_hip.npz)

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() {
    log "ОШИБКА: $*"
    exit 1
}

# Git Bash на Windows переписывает пути вида /app в C:/Program Files/Git/app - отключаем.
export MSYS_NO_PATHCONV=1

compose() {
    local files=(-f "$ROOT/docker-compose.yml")
    if [[ "${USE_GPU:-0}" == 1 ]]; then files+=(-f "$ROOT/docker-compose.gpu.yml"); fi
    docker compose --project-directory "$ROOT" "${files[@]}" "$@"
}

require_docker() {
    command -v docker >/dev/null || die "docker не найден"
    docker info >/dev/null 2>&1 || die "Docker не запущен (запустите Docker Desktop / dockerd)"
}

require_weights() {
    local dir="${1:-$WEIGHTS_DIR}" f
    for f in "${REQUIRED_WEIGHTS[@]}"; do
        [[ -f "$dir/$f" ]] || die "нет $dir/$f - см. README.md, раздел 2"
    done
    [[ -d "$dir/backbones" ]] || die "нет $dir/backbones"
}

# Абсолютный путь к существующему файлу/каталогу (docker -v требует абсолютные пути).
abspath() {
    [[ -e "$1" ]] || die "не найден путь: $1"
    if [[ -d "$1" ]]; then
        (cd "$1" && _pwd)
    else
        echo "$(cd "$(dirname "$1")" && _pwd)/$(basename "$1")"
    fi
}

# Флаги docker run: на Linux контейнер пишет в примонтированные каталоги от имени текущего
# пользователя; с USE_GPU=1 получает видеокарту.
RUN_FLAGS=()
if [[ "$(uname -s)" == Linux ]]; then RUN_FLAGS+=(--user "$(id -u):$(id -g)"); fi

enable_gpu() {
    USE_GPU=1
    RUN_FLAGS+=(--gpus all -e DXA_QC_DEVICE=cuda)
    # CUDA-сборка из docker-compose.gpu.yml, если образ не задан явно.
    # shellcheck disable=SC2034  # используется подключающими скриптами
    BACKEND_IMAGE="${DXA_QC_BACKEND_IMAGE:-dxa-qc-backend:gpu}"
}
