#!/usr/bin/env bash
# Пакетная обработка без веб-интерфейса, в изолированном контейнере без сети.
#   scripts/run_batch.sh <вход: файл|папка|zip> <папка для отчёта> [--format xlsx|csv] [--series] [--gpu]
# --series: ещё results_details.csv, PNG-визуализации и DICOM-серии (SR + Secondary Capture).
# Образ: DXA_QC_BACKEND_IMAGE (по умолчанию dxa-qc-backend:latest); для dxa-qc-backend:baked
# веса берутся из образа и не монтируются.
source "$(dirname "$0")/common.sh"

if [[ $# -lt 2 ]]; then
    sed -n '2,6p' "$0"
    exit 2
fi
input="$(abspath "$1")"
mkdir -p "$2"
output="$(abspath "$2")"
shift 2
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) enable_gpu ;;
        --format) extra+=(--format "${2:?--format требует значение}"); shift ;;
        --series) extra+=(--series) ;;
        *) die "неизвестный аргумент: $1" ;;
    esac
    shift
done

require_docker
name="$(basename "$input")"
# ZIP распаковывается на диск хоста, а не в tmpfs: архив закрытого набора может не поместиться в память.
work="$(mktemp -d "${TMPDIR:-/tmp}/dxa-qc-batch.XXXXXX")"
trap 'rm -rf "$work"' EXIT
mounts=(-v "$input:/input/$name:ro" -v "$output:/output" -v "$(abspath "$work"):/work")
if [[ "$BACKEND_IMAGE" != *:baked ]]; then
    require_weights
    mounts+=(-v "$WEIGHTS_DIR:/app/weights:ro")
fi

docker run --rm --network none --read-only --tmpfs /tmp:size=256m -e TMPDIR=/work \
    --security-opt no-new-privileges:true --cap-drop ALL "${RUN_FLAGS[@]}" \
    "${mounts[@]}" "$BACKEND_IMAGE" \
    python -m dxa_qc.cli.batch_process --input "/input/$name" --output /output "${extra[@]}"
log "отчёт: $output"
