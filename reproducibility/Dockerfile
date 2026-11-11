# Reproducible CPU image for StreamDIAM. Pinned base + pinned deps (R1 O18).
# Build:  docker build -t streamdiam -f reproducibility/Dockerfile .
# Run:    docker run --rm -e STREAMDIAM_ELLIPTIC_ROOT=/data/elliptic \
#                 -v /path/to/elliptic:/data/elliptic -v $PWD/results:/app/results \
#                 streamdiam pytest -q
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first for layer caching. CPU torch wheel keeps the image small;
# swap for a CUDA wheel if GPU training is wanted.
RUN pip install \
    numpy==2.5.3 \
    scipy==1.18.1 \
    scikit-learn==1.9.1 \
    matplotlib==3.11.2 \
    torch==2.15.0 \
    pytest pyflakes mypy

COPY . /app
ENV PYTHONPATH=/app/src

# Default: run the test suite (dataset-dependent tests skip without the data).
CMD ["pytest", "-q"]
