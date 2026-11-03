# NetCap API / pipeline image. No TensorFlow (the API never imports it). Data is mounted, never baked in.
#   docker build -f infra/Dockerfile -t netcap:1.0 .
FROM python:3.13.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    NETCAP_ROOT=/app NETCAP_DATA=/data NETCAP_RAW_ABILENE=/raw/abilene NETCAP_RAW_CESNET=/raw/cesnet

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY netcap/ netcap/
COPY netcap_app/ netcap_app/
COPY results/netcap/ results/netcap/
COPY docs/netcap/ docs/netcap/
COPY pyproject.toml .

RUN useradd --create-home --uid 10001 netcap && mkdir -p /data && chown -R netcap /app /data
USER netcap

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "netcap_app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
