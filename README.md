# DXA QC - контроль качества денситометрических исследований

DXA QC проверяет DICOM-снимки двухэнергетической рентгеновской денситометрии поясничного отдела
позвоночника и проксимального отдела бедра. По каждому снимку сервис определяет анатомическую
область и проекцию, оценивает качество укладки и разметки, называет тип нарушения и предлагает
коррекцию. Итоговое решение принимает специалист: он подтверждает или исправляет результат в
интерфейсе.

Решение работает в закрытом контуре. Сервис не обращается во внешние системы, снимки не покидают
хост.

## Материалы решения

- Презентация: [PDF](docs/presentation/Киберпсихоз_презентация_ЛЦТ2026.pdf) · [PPTX](docs/presentation/Киберпсихоз_презентация_ЛЦТ2026.pptx)
- Готовые веса моделей: [Releases v1.0](https://github.com/lolasudo/dxa-qc/releases/tag/v1.0)
- Команда «Киберпсихоз»: Семененко Лилия (капитан; DevOps, бэкенд, ML), Витютнев Олег (full-stack, фронтенд, дизайн)

## Состав решения

```
браузер --> frontend (nginx, React) --> REST API (FastAPI) --+
                                                              +--> ядро dxa_qc --> веса моделей (read-only)
CLI пакетной обработки ---------------------------------------+    (DICOM, модели, отчёты, DICOM-серии)
```

| Компонент | Назначение | Документация |
|---|---|---|
| `backend/` | API, пакетная обработка, модели, обучение | [backend/README.md](backend/README.md) |
| `frontend/` | веб-интерфейс, reverse proxy, заголовки безопасности | [frontend/README.md](frontend/README.md) |
| `scripts/` | сборка, запуск, офлайн-поставка, обучение | `--help` у каждого скрипта |
| `docs/` | эксплуатационная документация | см. ниже |

| Документ | Для кого |
|---|---|
| [USER_GUIDE](docs/USER_GUIDE.md) | врач, рентгенолаборант |
| [DEPLOYMENT_GUIDE](docs/DEPLOYMENT_GUIDE.md) | администратор: офлайн-поставка, GPU, пароль и HTTPS, лимиты, обновление |
| [TRAINING_GUIDE](docs/TRAINING_GUIDE.md) | ML-инженер: обучение, дообучение, формат манифеста |
| [SECURITY](docs/SECURITY.md) | служба ИБ: меры защиты, результаты аудита, принятые риски |
| [DEMO_SCENARIO](docs/DEMO_SCENARIO.md) | сценарий демонстрации |

## Требования к среде

| | Минимум | Рекомендуется |
|---|---|---|
| ОС | Linux x86-64; macOS или Windows с Docker Desktop | Linux x86-64 |
| Docker | 24+ с Compose v2 и Buildx | то же |
| CPU / RAM | 4 ядра, 4 ГБ (пик процесса около 0,9 ГБ) | 8 ядер, 8 ГБ |
| GPU | не требуется | NVIDIA от 4 ГБ, драйвер 570+, NVIDIA Container Toolkit |
| Диск | 10 ГБ: CPU-образ 2,3 ГБ, веса 0,3 ГБ, задачи | 30 ГБ; GPU-образ ещё 12 ГБ |

Производительность: около 0,5 с на снимок на CPU и 0,07 с на GPU. Для разработки дополнительно
нужны Node.js 22 LTS и Python 3.11 или 3.12.

На Windows скрипты `scripts/*.sh` запускайте из Git Bash.

## Быстрый запуск

Клонируйте репозиторий; дальнейшие команды выполняются из его корня:

```bash
git clone https://github.com/lolasudo/dxa-qc.git
cd dxa-qc
```

1. Скачайте веса со страницы [Releases v1.0](https://github.com/lolasudo/dxa-qc/releases/tag/v1.0), положите оба файла в корень репозитория и распакуйте в `backend/weights` (или обучите модели, см. раздел «Обучение»):

   ```bash
   curl -L -O https://github.com/lolasudo/dxa-qc/releases/download/v1.0/dxa-qc-weights-20260927.tar.gz
   curl -L -O https://github.com/lolasudo/dxa-qc/releases/download/v1.0/dxa-qc-weights-20260927.tar.gz.sha256
   sha256sum -c dxa-qc-weights-*.tar.gz.sha256
   tar -xzf dxa-qc-weights-*.tar.gz -C backend/
   ```

2. Соберите и запустите сервис:

   ```bash
   npm start            # сборка образов, запуск в фоне, ожидание готовности
   ```

   Интерфейс: http://localhost:7077. Первая сборка скачивает зависимости (в том числе torch) и
   зависит от скорости сети; повторные используют кэш Docker.

Эквивалентные варианты: `docker compose up --build` (логи в текущем окне) или `scripts/run.sh`.
GPU: `npm run start:gpu` или `scripts/run.sh --gpu`. Порт задаётся переменной `DXA_QC_PORT`.
Остановка: `npm stop`; задачи и отчёты при этом сохраняются, `scripts/stop.sh --purge` удаляет и их.

## Режимы работы

### Веб-интерфейс

Загрузка ZIP-архива или отдельного DICOM, таблица результатов, карточка снимка с разметкой,
заключением и предложением по коррекции, подтверждение или исправление результата специалистом,
выгрузка архива с отчётами и DICOM-сериями. Порядок работы описан в [USER_GUIDE](docs/USER_GUIDE.md).

### Пакетная обработка

Контейнер запускается без сети, вход монтируется только на чтение:

```bash
scripts/run_batch.sh <DICOM | папка | ZIP> <каталог результатов> [--format xlsx|csv] [--series] [--gpu]
```

Результат - `results.xlsx` (одна строка на снимок) и `results_errors.csv` для необработанных
файлов. С `--series` добавляются `results_details.csv` (заключения, измерения, коррекции),
PNG-визуализации и DICOM-объекты: SR с заключением и снимок с разметкой. Форматы описаны в
[backend/README.md](backend/README.md#форматы-данных).

### REST API

Префикс `/api/v1`; описание эндпоинтов - в [backend/README.md](backend/README.md#rest-api).

## Разработка

```bash
npm run dev      # API на :8000 с перезапуском, интерфейс на :5173 с горячей перезагрузкой
npm test         # ruff, pytest, eslint, tsc, vitest
```

Первый запуск `npm run dev` сам создаёт `backend/.venv` (torch с CUDA при наличии `nvidia-smi`) и
ставит зависимости фронтенда. Отдельно подготовить окружение: `npm run setup`.

## Обучение

Нужен датасет организатора (`Исследования/` и `разметка.xlsx`) или собственный манифест. Бэкбоны
скачиваются из сети один раз.

```bash
scripts/build.sh
scripts/train.sh --data Датасет --output backend/weights --download-backbones
```

Дообучение на данных клиники и на решениях специалистов описано в
[TRAINING_GUIDE](docs/TRAINING_GUIDE.md).

## Безопасность и доступ

- Backend подключён только к внутренней сети Docker. Автономность проверяется командой
  `scripts/check_offline.sh`: API и пакетная обработка запускаются в контейнере с `--network none`.
- По умолчанию интерфейс доступен всем, кто видит порт: сервис рассчитан на закрытый контур. Вход по
  паролю включается командой `scripts/set_password.sh <логин>` и выключается `--off`. Без HTTPS
  пароль передаётся открыто, см. [DEPLOYMENT_GUIDE](docs/DEPLOYMENT_GUIDE.md#пароль-и-https).
- Обращения к данным пациентов пишутся в журнал аудита. Подробности - в [SECURITY](docs/SECURITY.md).

## Команды

| Команда | Назначение |
|---|---|
| `npm start`, `npm run start:gpu` | собрать и запустить сервис, дождаться готовности |
| `npm stop`, `npm run logs` | остановить сервис, смотреть логи |
| `npm run dev`, `npm run setup`, `npm test` | разработка, подготовка окружения, все проверки |
| `scripts/build.sh [--gpu] [--bake-weights]` | собрать образы; `--bake-weights` - образ `dxa-qc-backend:baked` с весами внутри |
| `scripts/run.sh [--gpu] [--no-build]`, `scripts/stop.sh [--purge]` | запуск и остановка на bash |
| `scripts/run_batch.sh <вход> <выход>` | пакетная обработка без сети |
| `scripts/check_offline.sh [<вход>]` | проверка работы без сети |
| `scripts/train.sh --data <каталог>` | обучение моделей в контейнере |
| `scripts/set_password.sh <логин> \| --off` | вход по паролю |
| `scripts/package_weights.sh [--images]` | упаковка весов и образов для офлайн-поставки |

## Диагностика

| Симптом | Действие |
|---|---|
| `нет backend/weights/...` | скачайте веса из [Releases](https://github.com/lolasudo/dxa-qc/releases/tag/v1.0) или обучите модели |
| контейнер не стал healthy | `npm run logs`; как правило, причина - отсутствие весов |
| `Cannot connect to the Docker daemon` | запустите Docker Desktop или `sudo systemctl start docker` |
| порт занят | `DXA_QC_PORT=9000 npm start`; для `npm run dev` освободите порты 8000 и 5173 |
| `нужен Python 3.11 или 3.12` | установите Python 3.11 (`apt install python3.11 python3.11-venv`) |
| `could not select device driver` при `--gpu` | установите NVIDIA Container Toolkit или запускайте на CPU |
| «Файл слишком большой» | лимит 2 ГБ, см. [DEPLOYMENT_GUIDE](docs/DEPLOYMENT_GUIDE.md#лимиты) |

Обработка ошибок на уровне отдельных файлов описана в
[backend/README.md](backend/README.md#обработка-ошибок).
