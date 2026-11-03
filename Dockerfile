# syntax=docker/dockerfile:1

FROM node:20-bookworm-slim AS frontend-build

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV NEXT_PUBLIC_USE_MOCK=false
ENV NEXT_PUBLIC_API_BASE=
ENV NEXT_PUBLIC_WS_URL=
RUN npm run build

FROM python:3.10.13-slim-bookworm AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY focusfleet/ /app/focusfleet/
COPY backend/ /app/backend/
COPY --from=frontend-build /build/out /app/out

COPY runtime-assets/ /app/runtime-assets/

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENV PYTHONPATH=/app/backend:/app
ENV FOCUSFLEET_DATA_DIR=/data
ENV FOCUSFLEET_STATIC_DIR=/app/out
ENV PORT=7860

EXPOSE 7860

CMD ["/app/docker-entrypoint.sh"]
