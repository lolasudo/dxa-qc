#!/usr/bin/env bash
# Остановка сервиса.
#   scripts/stop.sh            остановить контейнеры (задания и отчёты сохраняются)
#   scripts/stop.sh --purge    также удалить том с заданиями
source "$(dirname "$0")/common.sh"

args=()
for arg in "$@"; do
    case "$arg" in
        --purge) args+=(--volumes) ;;
        -h | --help) sed -n '2,4p' "$0"; exit 0 ;;
        *) die "неизвестный аргумент: $arg" ;;
    esac
done
require_docker
compose down "${args[@]}"
log "остановлено"
