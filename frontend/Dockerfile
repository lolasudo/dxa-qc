# Build stage
# Базовые образы закреплены по дайджесту, обновление - через Dependabot или вручную.
FROM node:22-alpine@sha256:0a7108bf6c7bf5de370ffb1a3ed6be93d405b43ff159f681a8d18c0e2bc2e402 AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# В продакшене фронт обычно ходит в API через nginx proxy.
ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

ARG VITE_POLL_INTERVAL_MS=3000
ENV VITE_POLL_INTERVAL_MS=${VITE_POLL_INTERVAL_MS}

ARG VITE_MAX_UPLOAD_SIZE_MB=2048
ENV VITE_MAX_UPLOAD_SIZE_MB=${VITE_MAX_UPLOAD_SIZE_MB}

RUN npm run build

# Runtime stage
FROM nginxinc/nginx-unprivileged:1.30-alpine@sha256:4714e0b1b2577eaa1a6131d07c958b67f0eb68e6d0521e90c6e5287db8cf0bc5

# Security fixes of Alpine packages published after the base image was built.
USER root
RUN apk upgrade --no-cache \
    && mkdir -p /etc/nginx/snippets /etc/nginx/auth.d \
    && (find / -xdev -perm /6000 -type f -exec chmod a-s {} + 2>/dev/null || true)
USER 101

COPY security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1

USER 101

CMD ["nginx", "-g", "daemon off;"]