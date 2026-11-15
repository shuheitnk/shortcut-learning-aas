# The environment the paper's results are computed in, and the only container
# definition (the dev container builds this file too). It holds the interpreter,
# the locked dependencies and the toolchain, not the source: the working tree is
# mounted, so outputs/ and vendor/ land on the host.

ARG PYTHON_VERSION=3.11.15
ARG UV_VERSION=0.12.1

# Both images are pinned by digest because a tag moves; the digest is what the
# build resolves and the tag only names it, so change the two together.
FROM ghcr.io/astral-sh/uv:${UV_VERSION}@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

COPY --from=uv /uv /uvx /usr/local/bin/

ARG USERNAME=researcher
ARG USER_UID=1000
ARG USER_GID=1000

# git: pflacco is pinned to a commit and run manifests record the commit.
# build-essential: any dependency without a manylinux wheel.
# fonts-liberation: metrically identical to Arial, the figures' fallback font.
#
# The Debian archive is read from a fixed snapshot so a rebuild cannot move glibc
# (build-essential pulls libc6-dev of its own version). Snapshot Release files are
# past Valid-Until, and snapshot.debian.org is slow often enough to retry.
ARG DEBIAN_SNAPSHOT=20261001T112700Z
RUN printf '%s\n' \
        'Types: deb' \
        "URIs: http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" \
        'Suites: bookworm bookworm-updates' \
        'Components: main' \
        'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
        'Check-Valid-Until: no' \
        '' \
        'Types: deb' \
        "URIs: http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}" \
        'Suites: bookworm-security' \
        'Components: main' \
        'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
        'Check-Valid-Until: no' \
        > /etc/apt/sources.list.d/debian.sources
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
        fonts-liberation \
        git \
        less \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# The mounted tree belongs to the host account; without this git refuses to read
# it and the commit silently drops out of every run manifest.
RUN git config --system --add safe.directory '*'

# A non-root user. On a Linux host VS Code rewrites the UID/GID to the host
# account's, so files in the mounted tree stay the developer's. The rewrite chowns
# $HOME, which is why the environment lives there; with UID 1000 it is usually a
# no-op.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# A login shell runs /etc/profile, which resets PATH and drops the ENV below;
# without this, `python` there would be the interpreter without the dependencies.
RUN echo "PATH=\"/home/${USERNAME}/.venv/bin:/home/${USERNAME}/.local/bin:\$PATH\"" \
    > /etc/profile.d/10-shortcut-learning-aas.sh

# Determinism: the BLAS backend reads these once at load time, and a
# multi-threaded reduction is order-dependent.
ENV PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

# The environment lives in the image, not the mounted tree, where it would be
# the host's (unusable from a Windows host). Under $HOME it survives the UID rewrite.
ENV HOME=/home/${USERNAME} \
    VIRTUAL_ENV=/home/${USERNAME}/.venv \
    UV_PROJECT_ENVIRONMENT=/home/${USERNAME}/.venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/home/${USERNAME}/.venv/bin:/home/${USERNAME}/.local/bin:${PATH}

USER ${USERNAME}

# From PyPI, so uv stays the only package manager in the image.
RUN uv tool install rust-just==1.58.0

# The locked dependencies only; the project itself is installed from the mounted
# tree by `uv sync --locked`, so an edit under src/ needs no rebuild. About 2.5 GB,
# mostly the CUDA build of torch that uv.lock pins on Linux; the cache mount lets a
# retry reuse the downloads and keeps them out of the layer.
COPY --chown=${USER_UID}:${USER_GID} pyproject.toml uv.lock /tmp/lock/
RUN --mount=type=cache,target=/home/researcher/.cache/uv,uid=1000,gid=1000 \
    uv sync --project /tmp/lock --locked --no-install-project \
    && rm -rf /tmp/lock

# Mount point for devcontainer.json's volume, so an empty volume inherits this
# user's ownership rather than root's.
RUN mkdir -p ${HOME}/.cache/uv

# The Node.js pyright runs on. Unpinned, nodeenv reads the newest release from the
# Node.js index and fails on a pre-release version string.
ENV PYRIGHT_PYTHON_NODE_VERSION=22.11.0

# pyright fetches its own Node.js on first run; do it here so the quality gate
# does not stall. Not fatal: no result depends on pyright. A failed fetch leaves a
# partial environment that pyright would reuse, so it is removed.
RUN pyright --version \
    && rm -rf ${HOME}/.cache/pyright-python/nodeenv/src \
              ${HOME}/.cache/pyright-python/nodeenv/include \
    || { rm -rf ${HOME}/.cache/pyright-python/nodeenv; \
         echo "pyright warm-up skipped; it will fetch Node.js on first use"; }

WORKDIR /home/${USERNAME}
