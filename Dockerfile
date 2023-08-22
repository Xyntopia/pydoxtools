# Dockerfile
# Uses multi-stage builds requiring Docker 17.05 or higher
# See https://docs.docker.com/develop/develop-images/multistage-build/

# ------------ python-base -------------------------------------------- #

# TODO: check if we can somehow get rid of the apt-get dependencies (opencv, ghostscript etc..)
# they take up a lot of space ...

ARG PROJECT_DIR="/project"
ARG PROJECT_NAME="pydoxtools"

# FROM python:3.8.10-slim as python-base
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime as python-base
# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 as python-base
# compatible with pytorch1.10.1cu113

ARG PROJECT_DIR

# python configuration
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    # paths
    # our own app directory
    VENV_DIR="${PROJECT_DIR}/.venv/" \
    # prepend poetry and venv to path
    PATH="${PROJECT_DIR}/.venv/bin:$PATH"
    #PATH="${PROJECT_DIR}/LIBS/bin:$PATH"\
    #PYTHONPATH="${PROJECT_DIR}/LIBS:$PYTHONPATH"

# ------------------------- building the app ----------------------------
FROM python-base as builder-base
# install build dependencies
# g++ is needed for hnswlib compilation
# git is needed for yfinance
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
    iputils-ping build-essential \
    htop byobu \
    curl g++ wget\
    git \
    python3-minimal python3-pip\
    && pip install -U pip \
    && apt-get clean autoclean \
    && apt-get autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/
    
# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
# ENV POETRY_VERSION=1.0.5
# RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python3
RUN pip install --no-cache-dir poetry

# for the CPU-only version (whic is much smaller):
#RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
#    && pip install pytorch-lightning
# if we are using the pytorch images, pytorch is already pre-installed and
# we want to leave out the dependencies introduced by poetry.lock which
# would increase the size of the image...
RUN pip install --no-cache-dir pytorch-lightning transformers


ARG PROJECT_DIR
# copy only the pyproject.toml and lockfile in order to make sure they are cached
COPY ./pyproject.toml ./poetry.lock ${PROJECT_DIR}/
WORKDIR ${PROJECT_DIR}/

# export requirements text for pip in order to create packages pywheels as a later stage
#RUN poetry config virtualenvs.create false \
#    poetry install
# export dependencies including dev deps
# RUN poetry export -E pytorch-docker -E     etl --dev \
#     --without-hashes --no-interaction --no-ansi -f requirements.txt -o requirements.txt \
#    && ls -l
ARG PROJECT_NAME
#RUN --mount=type=cache,target=/root/.cache/ \ # if cache is required
# we are using poetry to install project in order to have pydoxtools symlinked
RUN --mount=type=cache,target=/root/.cache/ \
    mkdir "${PROJECT_DIR}/${PROJECT_NAME}" &&\
    touch "${PROJECT_DIR}/${PROJECT_NAME}/__init__.py" && \
    # poetry config virtualenvs.create false && \
    POETRY_VIRTUALENVS_CREATE=false \
    poetry install -E pytorch-docker -E etl -vvv
    #POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --no-dev -vvv

# this line needs to be run as:
#  >> COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose build doxcavator-serverless
# as we are caching the pip downloads with the --mount option which is only supported by docker with the
# above flags
# RUN --mount=type=cache,target=/root/.cache/pip pip install -I -t LIBS -r requirements.txt
# RUN --mount=type=cache,target=/root/.cache/pip pip install -t LIBS -r requirements.txt
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt


# -------------------------- production build --------------------------------------
# `production` image used for runtime
# FROM python-base as production
# if we need to install some apt packages....
#RUN apt-get update && \
#    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
#    #cleaning up
#    && apt-get clean autoclean \
#    && apt-get autoremove --yes \
#    && rm -rf /var/lib/{apt,dpkg,cache,log}/

ARG PROJECT_DIR

# install rclone in order to synchronize training files
RUN curl https://rclone.org/install.sh | bash

# TODO: copy training datasets
# TODO: copy training datasets from online repository (S3)?
COPY ./training_data "${PROJECT_DIR}/training_data"
# COPY ./src directory here
# COPY --from=builder-base $VENV_DIR $VENV_DIR
# COPY --from=builder-base "$PROJECT_DIR/LIBS" "$PROJECT_DIR/LIBS"
# COPY --from=builder-base "/usr/lib" "/usr/lib"
# COPY --from=builder-base "/opt/conda" "/opt/conda"
# COPY --from=builder-base "$PROJECT_DIR/requirements.txt" "$PROJECT_DIR/requirements.txt"
# WORKDIR ${PROJECT_DIR}/
# RUN pip install --no-cache-dir -r requirements.txt
# ENV PATH="$PROJECT_DIR/LIBS:$PATH"

ARG PROJECT_NAME
# copy only necessary files
COPY ./analysis      ${PROJECT_DIR}/analysis/
COPY ./$PROJECT_NAME ${PROJECT_DIR}/$PROJECT_NAME
# initialize training cache
# this is no needed right now, as the calculation was sped up siginificantly
# due to multiprocessing
# RUN python -c 'from pydoxtools import classifier; classifier.load_labeled_text_blocks()'

#RUN mkdir $HOME/comcharax
WORKDIR "${PROJECT_DIR}"
#jupyter lab --allow-root --ip 0.0.0.0 --no-browser
# pre-caching textblocks...
# RUN python -c "from pydoxtools import classifier; classifier.load_labeled_text_blocks()"
# enable parallel transformer tokenizers
ENV TOKENIZERS_PARALLELISM=true
ENTRYPOINT [ "jupyter", "lab", "--allow-root", "--ip", "0.0.0.0", "--no-browser" ]
#CMD ["uvicorn", "comcharax_restfulapi:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "info"]

# -------------------------- pip install test --------------------------------------

FROM python:3.10-slim as test3.10

# TODO: add cachin directories for models, spacy etc...

#POETRY_CACHE_DIR=/tmp/poetry_cache
#--mount=type=cache,target=$POETRY_CACHE_DIR

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
    iputils-ping build-essential \
    htop byobu \
    curl g++ wget\
    git file  \
    tesseract-ocr tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-spa \
    poppler-utils graphviz graphviz-dev\
    && pip install -U pip \
    && apt-get clean autoclean \
    && apt-get autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN wget https://github.com/jgm/pandoc/releases/download/2.19.2/pandoc-2.19.2-1-amd64.deb
RUN dpkg -i pandoc-2.19.2-1-amd64.deb

RUN pip install pytest pygraphviz

# ------------------------- run test for pipy/github installation --------------------------------------

FROM test3.10 as test_remote
#RUN pip install pydoxtools
# install from github project itself
RUN --mount=type=cache,target=/root/.cache \
    pip install -U "pydoxtools[etl,inference] @ git+https://github.com/xyntopia/pydoxtools.git@python3.8_support"
RUN git clone --recurse-submodules -b python3.8_support https://github.com/xyntopia/pydoxtools.git

# -------------------------- run tests from local installation --------------------------------------

FROM test3.10 as test_local

COPY . /pydoxtools/
WORKDIR pydoxtools
RUN --mount=type=cache,target=/root/.cache  \
    pip install ".[etl,inference]"
#RUN pytest
