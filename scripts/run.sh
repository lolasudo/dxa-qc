#!/usr/bin/env bash
# Запуск сервиса и ожидание готовности моделей.
#   scripts/run.sh [--gpu] [--no-build]
# Порт: DXA_QC_PORT (по умолчанию 7077).
source "$(dirname "$0")/common.sh"

BUILD=(--build)
for arg in "$@"; do
    case "$arg" in
        --gpu) enable_gpu ;;
        --no-build) BUILD=() ;;
        -h | --help) sed -n '2,4p' "$0"; exit 0 ;;
        *) die "неизвестный аргумент: $arg" ;;
    esac
done

require_docker
require_weights
compose up -d "${BUILD[@]}"

port="${DXA_QC_PORT:-7077}"
# DXA_QC_PORT - порт или адрес:порт (например 127.0.0.1:7077 за HTTPS-прокси).
if [[ "$port" == *:* ]]; then base="http://$port"; else base="http://localhost:$port"; fi
url="$base/api/v1/health"
log "ожидание готовности моделей ($url)"
for _ in $(seq 1 90); do
    if curl -fsS "$url" 2>/dev/null | grep -Eq '"models_ready": ?true'; then
        log "сервис готов: $base"
        exit 0
    fi
    sleep 2
done
compose ps
compose logs --tail 50 backend
die "сервис не стал готов за 3 минуты (логи выше)"
