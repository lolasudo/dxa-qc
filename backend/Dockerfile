# DXA QC backend: REST API + batch CLI. One image serves both.
#
#   docker build -t dxa-qc-backend .                                   # CPU torch (~2.3 GB on disk)
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t dxa-qc-backend .  # GPU (H200)
#
# Model weights are NOT baked in by default: mount them read-only at /app/weights
# (see ../README.md). They are produced by `python -m dxa_qc.training.*`.
# Self-contained image for offline delivery (weights inside, nothing to mount):
#
#   docker buildx build --target baked --build-context weights=./weights -t dxa-qc-backend:baked --load .

# Base image pinned by digest.
FROM python:3.14-slim-bookworm@sha256:82bc3c539b8813ada9d68c63b40158fa002f7f33de9bf3312a3dfdc0620dff56 AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DXA_QC_CONFIG_DIR=/app/configs \
    DXA_QC_WEIGHTS_DIR=/app/weights \
    DXA_QC_DATA_DIR=/data/jobs \
    TMPDIR=/data/tmp

WORKDIR /app

# torch + torchvision first (largest layer, changes rarely), both from TORCH_INDEX_URL: otherwise
# timm pulls torchvision from PyPI together with ~2.5 GB of CUDA libraries even for a CPU image.
RUN pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
# Exact versions from the lock file, except torch (platform-specific build) and dev-only tools.
COPY requirements-dev.lock ./
RUN grep -vE '^(#|torch|torchvision|pytest|ruff|iniconfig|pluggy|Pygments)' requirements-dev.lock > /tmp/constraints.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -c /tmp/constraints.txt ".[ml,api,train]" && rm /tmp/constraints.txt

# The service never installs packages: drop the packaging toolchain (and the CVEs of pip, its
# vendored libraries and setuptools that trivy reports) from the runtime image. Also drop the build
# leftovers and the source copy (the package lives in site-packages) and the setuid/setgid bits,
# which nothing in the service needs (CIS Docker 4.8).
RUN pip uninstall -y setuptools wheel \
    && python -m pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.11/ensurepip /app/build /app/src /app/*.egg-info \
    && (find / -xdev -perm /6000 -type f -exec chmod a-s {} + 2>/dev/null || true)

COPY configs ./configs

# Non-root user; only /data (job files, temp extraction) is writable.
RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin dxaqc \
    && mkdir -p /app/weights /data/jobs /data/tmp \
    && chown -R dxaqc:dxaqc /data
USER dxaqc

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import json,urllib.request,sys; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)); sys.exit(0 if d.get('models_ready') else 1)"

CMD ["python", "-m", "dxa_qc.api", "--host", "0.0.0.0", "--port", "8000"]

# ---- weights baked in: the image alone runs the service (scripts/build.sh --bake-weights) ----
FROM runtime AS baked
COPY --from=weights *.npz *.json /app/weights/
COPY --from=weights backbones/ /app/weights/backbones/

# ---- default target: weights mounted at runtime ----
FROM runtime
