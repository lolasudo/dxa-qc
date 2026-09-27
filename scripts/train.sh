#!/usr/bin/env bash
# Обучение моделей в контейнере backend (тот же код и те же версии библиотек, что в сервисе).
#
#   scripts/train.sh --data <корень датасета> [--manifest <csv>] [--config <yaml>]
#                    [--output <папка весов>] [--reps N] [--download-backbones] [--gpu]
#
# --data      каталог с изображениями; пути манифеста считаются от него.
# --manifest  манифест обучения (формат: backend/src/dxa_qc/data/manifest.py). Без него --data
#             должен быть в формате организатора: Исследования/ + разметка.xlsx; манифест
#             строится автоматически по backend/data/region_annotations.csv.
# --output    куда записать новые веса (по умолчанию backend/weights.new; рабочие веса
#             backend/weights не трогаются - замените их после проверки метрик).
# --download-backbones  скачать бэкбоны из конфига (нужна сеть); иначе копируются из backend/weights.
# Эмбеддинги кэшируются в backend/var/embedding_cache: повторное обучение быстрое.
source "$(dirname "$0")/common.sh"

data="" manifest="" config="" output="$ROOT/backend/weights.new" download=0
train_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) data="${2:?}"; shift ;;
        --manifest) manifest="${2:?}"; shift ;;
        --config) config="${2:?}"; shift ;;
        --output) output="${2:?}"; shift ;;
        --reps) train_args+=(--reps "${2:?}"); shift ;;
        --download-backbones) download=1 ;;
        --gpu) enable_gpu ;;
        -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
        *) die "неизвестный аргумент: $1" ;;
    esac
    shift
done
[[ -n "$data" ]] || die "укажите --data (см. --help)"
data="$(abspath "$data")"
require_docker

mkdir -p "$output/backbones"
output="$(abspath "$output")"
mkdir -p "$ROOT/backend/var"
mounts=(-v "$data:/dataset:ro" -v "$output:/out" -v "$(abspath "$ROOT/backend/var"):/work/var")
if [[ -n "$manifest" ]]; then
    manifest="$(abspath "$manifest")"
    mounts+=(-v "$manifest:/in/manifest.csv:ro")
    manifest_in=/in/manifest.csv
else
    [[ -d "$data/Исследования" && -f "$data/разметка.xlsx" ]] ||
        die "без --manifest нужен формат организатора: $data/Исследования и $data/разметка.xlsx"
    mounts+=(-v "$ROOT/backend/data:/in/data:ro")
    manifest_in=""
fi
if [[ -n "$config" ]]; then
    mounts+=(-v "$(abspath "$config"):/in/training.yaml:ro")
    train_args+=(--config /in/training.yaml)
fi

if [[ "$download" == 0 ]]; then
    require_weights
    log "копирование бэкбонов из $WEIGHTS_DIR/backbones"
    cp -r "$WEIGHTS_DIR/backbones/." "$output/backbones/"
fi

# Шаги внутри контейнера; пути только контейнерные, аргументы передаются через окружение.
# shellcheck disable=SC2016  # переменные раскрывает sh внутри контейнера
steps='
set -eu
cd /work
if [ -n "$MANIFEST_IN" ]; then
    cp "$MANIFEST_IN" /out/manifest.csv
else
    python -m dxa_qc.training.make_manifest --studies /dataset/Исследования \
        --labels /dataset/разметка.xlsx --annotations /in/data/region_annotations.csv \
        --output /out/manifest.csv --images-root /dataset
fi
if [ "$DOWNLOAD" = 1 ]; then
    python -m dxa_qc.training.backbones --weights-dir /out/backbones $CONFIG_ARG
fi
python -m dxa_qc.training.region --manifest /out/manifest.csv --images-root /dataset \
    --output /out/region_classifier.npz
python -m dxa_qc.training.quality --manifest /out/manifest.csv --images-root /dataset \
    --weights-dir /out "$@"
'
config_arg=""
if [[ -n "$config" ]]; then config_arg="--config /in/training.yaml"; fi

log "обучение: данные $data -> веса $output"
docker run --rm --security-opt no-new-privileges:true --cap-drop ALL "${RUN_FLAGS[@]}" \
    -e MANIFEST_IN="$manifest_in" -e DOWNLOAD="$download" -e CONFIG_ARG="$config_arg" \
    -e HF_HOME=/tmp/hf -e TORCH_HOME=/tmp/torch \
    "${mounts[@]}" "$BACKEND_IMAGE" sh -c "$steps" train "${train_args[@]}"
log "готово. Метрики: $output/quality_metrics.json"
log "после проверки: замените backend/weights содержимым $output и перезапустите сервис"
