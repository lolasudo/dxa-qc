#!/usr/bin/env bash
# Проверка автономности: сервис стартует и обрабатывает исследования без доступа к сети.
#   scripts/check_offline.sh [<вход для пробной обработки>]
# Вход по умолчанию - «Датасет/Для теста.zip» (или папка, если архива нет). Образ: DXA_QC_BACKEND_IMAGE.
source "$(dirname "$0")/common.sh"

default="$ROOT/Датасет/Для теста.zip"
[[ -f "$default" ]] || default="$ROOT/Датасет/Для теста"
sample="${1:-$default}"
container=dxa-qc-offline-check
require_docker
mounts=()
if [[ "$BACKEND_IMAGE" != *:baked ]]; then
    require_weights
    mounts=(-v "$WEIGHTS_DIR:/app/weights:ro")
fi
out="$(mktemp -d)"
trap 'rm -rf "$out"; docker rm -f "$container" >/dev/null 2>&1 || true' EXIT

log "1/2 API в контейнере с --network none: ожидание готовности моделей"
docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --network none --read-only --tmpfs /tmp:size=256m \
    --tmpfs /data:size=1g,mode=1777 -e TMPDIR=/data "${mounts[@]}" "$BACKEND_IMAGE" >/dev/null
state=starting
for _ in $(seq 1 90); do
    state="$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null || echo gone)"
    [[ "$state" == starting ]] || break
    sleep 2
done
if [[ "$state" != healthy ]]; then
    docker logs --tail 50 "$container"
    die "API не стал готов без сети (статус: $state)"
fi
log "API готов без сети"

log "2/2 пакетная обработка без сети: $sample"
"$ROOT/scripts/run_batch.sh" "$sample" "$out" --format csv
[[ -s "$out/results.csv" ]] || die "отчёт не создан"
rows=$(($(wc -l <"$out/results.csv") - 1))
log "отчёт создан: $rows строк"
log "ПРОВЕРКА АВТОНОМНОСТИ ПРОЙДЕНА"
