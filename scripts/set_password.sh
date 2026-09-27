#!/usr/bin/env bash
# Вход в веб-интерфейс и API по паролю (HTTP Basic через nginx). По умолчанию вход открыт:
# сервис рассчитан на закрытый контур, и автоматическая проверка не должна упираться в пароль.
#   scripts/set_password.sh <логин>      задать/сменить пароль (спросит его; или DXA_QC_PASSWORD)
#   scripts/set_password.sh --off        отключить вход по паролю
# Применяется после перезапуска: scripts/run.sh --no-build. Пароль хранится только в виде хеша.
# Без TLS пароль передаётся открыто - в сети используйте HTTPS-прокси (docs/DEPLOYMENT_GUIDE.md).
source "$(dirname "$0")/common.sh"

dir="$ROOT/deploy/auth"
mkdir -p "$dir"
case "${1:-}" in
    "" | -h | --help) sed -n '2,7p' "$0"; exit 0 ;;
    --off)
        rm -f "$dir/auth.conf" "$dir/htpasswd"
        log "вход по паролю отключён; перезапустите: scripts/run.sh --no-build"
        exit 0
        ;;
esac
user="$1"
[[ "$user" =~ ^[A-Za-z0-9._-]{1,32}$ ]] || die "логин: латиница, цифры, . _ - (до 32 символов)"
command -v openssl >/dev/null || die "нужен openssl"
password="${DXA_QC_PASSWORD:-}"
if [[ -z "$password" ]]; then
    read -r -s -p "Пароль для $user: " password; echo >&2
    read -r -s -p "Повторите: " again; echo >&2
    [[ "$password" == "$again" ]] || die "пароли не совпадают"
fi
[[ ${#password} -ge 10 ]] || die "пароль короче 10 символов"

hash="$(printf '%s' "$password" | openssl passwd -apr1 -stdin)"
tmp="$(mktemp)"
if [[ -f "$dir/htpasswd" ]]; then grep -v "^$user:" "$dir/htpasswd" >"$tmp" || true; fi
echo "$user:$hash" >>"$tmp"
mv "$tmp" "$dir/htpasswd"
chmod 644 "$dir/htpasswd"
printf 'auth_basic "DXA QC";\nauth_basic_user_file /etc/nginx/auth.d/htpasswd;\n' >"$dir/auth.conf"
log "пароль для $user сохранён; перезапустите: scripts/run.sh --no-build"
