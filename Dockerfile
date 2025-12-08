# --- Stage 1: Build Stockfish (rarely changes, cached longer) ---
# Platform note: CI/CD and Modal use linux/amd64. Local dev on Apple Silicon uses linux/arm64.
# Both platforms supported via torchvision 0.23.0 (see pyproject.toml for compatibility notes).
# To force a specific platform: docker build --platform=linux/amd64 ...
FROM python:3.13-slim-bookworm AS stockfish-builder

ARG VERSION=17.1
ARG BUILD_PROFILE=native
WORKDIR /chess

RUN apt-get update \
    && apt-get install -y wget build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget "https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_$VERSION.tar.gz" \
    && tar -xf "sf_$VERSION.tar.gz" -C /chess \
    && rm *.gz

WORKDIR /chess/Stockfish-sf_$VERSION/src
RUN make net && make build ARCH=$BUILD_PROFILE

# Move compiled binary to standard location
RUN mv stockfish /usr/local/bin/stockfish

# --- Stage 1b: Build lc0 (Leela Chess Zero) for Maia ---
FROM python:3.13-slim-bookworm AS lc0-builder

ARG LC0_VERSION=0.32
WORKDIR /chess

RUN apt-get update \
    && apt-get install -y git python3-pip g++ ninja-build zlib1g-dev libopenblas-dev \
    && pip3 install --break-system-packages meson \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b release/$LC0_VERSION --depth 1 https://github.com/LeelaChessZero/lc0.git

WORKDIR /chess/lc0
RUN ./build.sh

# Move compiled binary to standard location
RUN mv build/release/lc0 /usr/local/bin/lc0

# --- Stage 2: Install Python Dependencies (changes more frequently) ---
FROM python:3.13-slim-bookworm AS python-deps

ENV PYTHONUNBUFFERED=True
WORKDIR /app

# Install uv (from linux/amd64 variant)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project definition files for caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# --- Stage 3: Final Application Image ---
FROM python:3.13-slim-bookworm AS runner

WORKDIR /app

# Install wget for downloading Maia weights and libopenblas for lc0 runtime
RUN apt-get update \
    && apt-get install -y wget libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from python-deps stage
COPY --from=python-deps /app/.venv ./.venv

# Copy Stockfish binary from stockfish-builder stage
COPY --from=stockfish-builder /usr/local/bin/stockfish /usr/local/bin/stockfish

# Copy lc0 binary from lc0-builder stage
COPY --from=lc0-builder /usr/local/bin/lc0 /usr/local/bin/lc0

# Download Maia weights (1100 rating model for human move prediction)
RUN mkdir -p /app/data \
    && wget -q -O /app/data/maia-1100.pb.gz \
       https://raw.githubusercontent.com/CSSLab/maia-chess/master/maia_weights/maia-1100.pb.gz

# Copy application code
COPY . /app

# Set environment variables for engine paths
ENV STOCKFISH_PATH=/usr/local/bin/stockfish
ENV LC0_PATH=/usr/local/bin/lc0
ENV MAIA_WEIGHTS_PATH=/app/data/maia-1100.pb.gz

# Set the entrypoint
CMD ["/app/.venv/bin/python", "/app/chess_sandbox/server.py"]
