# Руководство по развёртыванию

Документ описывает развёртывание в клинической сети: без интернета, на CPU или GPU, с обновлением
и мониторингом. Базовый запуск описан в [корневом README](../README.md).

## Архитектура

```
браузер ──► frontend (nginx, :7077) ──/api/──► backend (FastAPI, :8000)
            сети: public + internal              сеть: только internal (нет выхода в интернет)
                                                 ├─ веса /app/weights (только чтение)
                                                 └─ задачи /data (том jobs, TTL 24 ч)
```

- **frontend** - nginx-unprivileged с собранным React-интерфейсом. Он проксирует `/api/` на backend,
  ограничивает частоту запросов и размер загрузки и отдаёт заголовки безопасности (CSP и др.).
- **backend** - подключён только к внутренней сети Docker (`internal: true`): он не публикуется
  наружу и сам не может обратиться в интернет. Один процесс uvicorn. Задачи выполняются в одном рабочем потоке по очереди: так
  потребление памяти предсказуемо и GPU не перегружается. Тот же образ используется для CLI и
  обучения.

## Варианты поставки

### А. Машина с доступом к Docker Hub и PyPI

```bash
tar -xzf dxa-qc-weights-*.tar.gz -C backend/    # или scripts/train.sh
npm start            # или docker compose up --build, или scripts/run.sh
npm run start:gpu    # NVIDIA GPU (или scripts/run.sh --gpu)
```

### Б. Закрытый контур без интернета

На машине с интернетом:

```bash
scripts/build.sh                     # или --gpu
scripts/package_weights.sh --images  # dist/: веса и образы + .sha256
```

Перенесите на целевую машину репозиторий (без `Датасет/`) и содержимое `dist/`, затем выполните:

```bash
cd dist && sha256sum -c *.sha256 && cd ..
docker load -i dist/dxa-qc-images-*.tar.gz
tar -xzf dist/dxa-qc-weights-*.tar.gz -C backend/
scripts/run.sh --no-build
scripts/check_offline.sh             # подтверждение работы без сети
```

### В. Один самодостаточный образ (только пакетная обработка)

`scripts/build.sh --bake-weights` собирает `dxa-qc-backend:baked` с весами внутри. Такому образу не
нужны ни репозиторий, ни монтирование весов:

```bash
docker run --rm --network none --read-only --tmpfs /tmp --tmpfs /data:mode=1777 \
    -v /input:/input:ro -v /output:/output dxa-qc-backend:baked \
    python -m dxa_qc.cli.batch_process --input /input --output /output --format xlsx
# или: DXA_QC_BACKEND_IMAGE=dxa-qc-backend:baked scripts/run_batch.sh /input /output
```

## GPU

Нужны драйвер NVIDIA 570+ (CUDA 12.8) и NVIDIA Container Toolkit. Проверка:
`docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.

`scripts/build.sh --gpu` и `scripts/run.sh --gpu` собирают CUDA-сборку torch и передают сервису
видеокарту (`docker-compose.gpu.yml`). На H200 используется одна GPU на экземпляр. Модели
лёгкие (4 замороженных бэкбона, < 1 ГБ видеопамяти), поэтому GPU ускоряет в основном прогрев и
большие пакеты. На CPU снимок тоже обрабатывается < 1 с.

Устройство выбирается переменной `DXA_QC_DEVICE` (`cuda`, `cuda:1`, `cpu`); по умолчанию - cuda при
наличии. Решения модели от устройства не зависят: эмбеддинги CPU и GPU совпадают с косинусом
> 0.999, и это проверяется тестом.

## Пароль и HTTPS

**Вход по паролю.** По умолчанию интерфейс и API доступны всем, кто видит порт: сервис рассчитан на
закрытый контур, и автоматическая проверка не должна упираться в пароль. Чтобы включить вход:

```bash
scripts/set_password.sh doctor          # спросит пароль; или DXA_QC_PASSWORD=... для автоматизации
scripts/set_password.sh nurse           # ещё пользователь - дописывается в тот же файл
npm stop && npm start                   # применить
scripts/set_password.sh --off           # отключить
```

Файлы лежат в `deploy/auth/` (в git не попадают) и монтируются в nginx только для чтения. Пароль
хранится как хеш `apr1`. `/healthz` и `/api/v1/health` доступны без пароля - по ним работают
проверки готовности.

**HTTPS.** Сервис не терминирует TLS сам. В сети больницы поставьте перед ним HTTPS-прокси
(корпоративный балансировщик, nginx или Caddy с сертификатом организации) и публикуйте наружу
только его, например так (фрагмент nginx на хосте):

```nginx
server {
  listen 443 ssl;
  server_name dxa-qc.clinic.local;
  ssl_certificate     /etc/ssl/clinic/dxa-qc.crt;
  ssl_certificate_key /etc/ssl/clinic/dxa-qc.key;
  add_header Strict-Transport-Security "max-age=31536000" always;
  client_max_body_size 2048m;
  location / { proxy_pass http://127.0.0.1:7077; proxy_request_buffering off; }
}
```

Порт 7077 в этом случае публикуйте только на localhost:
`DXA_QC_PORT=127.0.0.1:7077 npm start`.

## Настройки

| Что | Где | По умолчанию |
|---|---|---|
| Внешний порт | `DXA_QC_PORT` | 7077 |
| Устройство | `DXA_QC_DEVICE` | авто |
| Память backend | `DXA_QC_BACKEND_MEMORY` | 8g |
| Swagger-документация API | `DXA_QC_API_DOCS=1` - показать | скрыта (в `npm run dev` показана) |
| CPU backend | `DXA_QC_BACKEND_CPUS` | 4 |
| Клинические пороги (ось 5°, отступы ROI) | `backend/configs/thresholds.yaml` | клинические нормы |
| Порог уверенности области | `thresholds.yaml: region.min_confidence` | 0.6 |
| Разделитель `violation_type` | `thresholds.yaml: report.violation_separator` | `;` |

Конфиги копируются в образ при сборке. После изменения выполните `scripts/run.sh` - он
пересоберёт образ.

### Лимиты

| Лимит | Где | По умолчанию | Ответ при превышении |
|---|---|---|---|
| Размер загрузки | `service.yaml: api.max_upload_bytes`, `frontend/nginx.conf: client_max_body_size`, `VITE_MAX_UPLOAD_SIZE_MB` - менять согласованно | 2 ГБ | 413 |
| Задач в очереди | `service.yaml: api.max_queued_jobs` | 8 | 429 |
| Одновременных `/study` | `service.yaml: api.max_pending_studies` | 4 | 429 |
| Хранилище задач / свободное место | `api.max_storage_bytes` / `api.min_free_disk_bytes` | 20 ГБ / 5 ГБ | 507 |
| Срок хранения задач | `api.job_ttl_hours` | 24 ч | - |
| Соединений uvicorn | `api.max_connections` | 64 | 503 |
| Частота запросов (nginx) | `frontend/nginx.conf` | API 20 r/s, загрузка 10 r/min на IP | 429/503 |
| Размер DICOM / изображения / архива | `thresholds.yaml: ingest.*` | 64 МБ / 4096 px / 8 ГБ, до 20 000 файлов | `Failure` / отказ архива |

## Эксплуатация

- **Готовность:** `GET /api/v1/health` -> `models_ready: true`. Docker healthcheck backend
  использует тот же endpoint, frontend стартует только после него.
- **Логи:** `docker compose logs -f backend` (stderr, структурированные строки с уровнем).
- **Данные пациентов:** загруженные файлы, отчёты и превью хранятся в томе `jobs` и удаляются через
  `job_ttl_hours`. `scripts/stop.sh --purge` удаляет их сразу. Сервис не делает внешних
  запросов (проверяется `scripts/check_offline.sh`), а CSP интерфейса запрещает внешние ресурсы.
  Для гарантии на уровне инфраструктуры запретите исходящий трафик хоста или сети Docker.
- **Обновление моделей:** обучите веса в `backend/weights.new` ([TRAINING_GUIDE](TRAINING_GUIDE.md)),
  сравните метрики, замените `backend/weights`, выполните `npm stop && npm start`. Откат -
  вернуть прежний каталог весов.
- **Обновление версии:** `git pull && npm start`. Задачи в томе сохраняются.
- **Журнал аудита:** `docker compose logs backend | grep dxa_qc.audit` - кто и когда загружал
  исследования, смотрел и скачивал результаты.
- **Масштабирование:** для большого потока запустите несколько экземпляров backend (по одному на
  GPU) за балансировщиком. Задачи экземпляра хранятся локально, поэтому нужна привязка клиента
  к экземпляру (sticky sessions) по `job_id`.

## Зафиксированные версии

Базовые образы `python:3.11-slim-bookworm`, `node:22-alpine`,
`nginxinc/nginx-unprivileged:1.30-alpine` закреплены по дайджесту `sha256` в Dockerfile. Python-зависимости - `backend/requirements-dev.lock`
(torch 2.11.0, torchvision 0.26.0 задаются в Dockerfile), npm - `frontend/package-lock.json`. Для
каждой сборки в CI формируется SBOM (CycloneDX).
