FROM python:3.13.7-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IE_CONFIG=/app/configs/tuned.yaml \
    IE_RUN_ROOT=/app/runs \
    IE_SWANLAB=1 \
    IE_SWANLAB_MODE=offline

WORKDIR /app

# 核心依赖严格锁定；openai / swanlab 单独一组区间约束，避免上游命名漂移导致构建失败。
#
# PIP_INDEX_URL 只影响构建期下载速度，不改变锁定版本：requirements.lock 是精确 pin 的，
# 换源只会换"从哪里取同一个版本"，不会换"取哪个版本"。默认保留官方源，
# 保证评测方在本机按原样构建时无需任何额外参数。
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.lock requirements-extras.txt ./
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements.lock \
    && pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements-extras.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-build-isolation --no-deps .

COPY configs ./configs
COPY run.sh ./run.sh

RUN chmod +x /app/run.sh \
    && useradd --create-home --uid 10001 runner \
    && mkdir -p /app/runs \
    && chown -R runner:runner /app
USER runner

# 运行契约：`docker run <image> [demo|csv|smoke|check|record]`，全部产物写入 IE_RUN_ROOT。
#
# 默认开启 SwanLab 记录但用 offline：评测机通常无外网，online 会降级，不如显式落盘——
# run 结束时 summary/metrics/events/注册表 都写进 $IE_RUN_ROOT/.swanlab，事后
# `swanlab sync` 可补传。要直连云端，设置 SWANLAB_API_KEY 并 IE_SWANLAB_MODE=online。
ENTRYPOINT ["/app/run.sh"]
CMD ["demo"]
