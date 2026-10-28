FROM python:3.12-slim
ARG EXTRAS=""
ENV PYTHONUNBUFFERED=1 HF_HOME=/data/hf
WORKDIR /opt/j7scope
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE NOTICE ./
COPY j7scope ./j7scope
COPY apps/serve/j7scope_serve ./apps/serve/j7scope_serve
COPY apps/site ./apps/site
COPY data/probe_prompts_en.jsonl data/probe_prompts_zh.jsonl ./data/
RUN pip install --no-cache-dir ".${EXTRAS}" \
    && useradd --uid 10001 --create-home researcher \
    && mkdir /data && chown researcher:researcher /data
USER researcher
WORKDIR /data
EXPOSE 8799
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8799/health',timeout=3)"
ENTRYPOINT ["j7scope"]
CMD ["--backend", "mock", "--host", "0.0.0.0", "--record", "/data/traces"]
