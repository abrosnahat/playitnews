# Image содержит только зависимости (python, ffmpeg, deno, pip-пакеты).
# Код, .env, data.db, токены и медиа монтируются с хоста (см. docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates fonts-dejavu-core fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# JS-рантайм для yt-dlp (n-challenge solving, yt-dlp-ejs)
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
