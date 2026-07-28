# kb-server: FastAPI app + Inbox/Log watchers, packaged per the design's
# Deployment section (Docker container, port 8090).
#
# Build context is the repo root: `docker build -f docker/Dockerfile .`

FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

FROM python:3.12-slim-trixie
RUN groupadd --system --gid 999 kb \
 && useradd --system --gid 999 --uid 999 --create-home kb

COPY --from=builder --chown=kb:kb /app /app
# HOME is set explicitly (not left to /etc/passwd's kb entry) because
# docker-compose.yml's `user:` override lets operators run this image as an
# arbitrary UID:GID to match their vault's ownership. An overridden UID has
# no /etc/passwd entry, so Path.home() (used by kb.core.config.load_config)
# would otherwise fall back to "/" and silently read/write the wrong config
# path instead of erroring loudly.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HOME=/home/kb

# kb-server reads ~/.config/kb/config.yaml for vault_path (kb.core.config);
# point it at the volume mounted by docker-compose.yml.
RUN mkdir -p /home/kb/.config/kb \
 && printf 'vault_path: /data/obsidian\nmachine_name: kb-host\n' \
    > /home/kb/.config/kb/config.yaml \
 && chown -R kb:kb /home/kb/.config \
 && chmod o+rx /home/kb

USER kb
WORKDIR /app
EXPOSE 8090
CMD ["kb-server"]
