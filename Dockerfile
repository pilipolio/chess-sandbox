# --- Stage 1: Build Stockfish versions (rarely changes, cached longer) ---
FROM python:3.13-slim-bookworm AS stockfish-builder

ARG VERSION=17.1
ARG VERSION_8=8
ARG BUILD_PROFILE=native
WORKDIR /chess

RUN apt-get update \
    && apt-get install -y wget build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Build Stockfish latest version
RUN wget "https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_$VERSION.tar.gz" \
    && tar -xf "sf_$VERSION.tar.gz" -C /chess \
    && rm *.gz

WORKDIR /chess/Stockfish-sf_$VERSION/src
RUN make net && make build ARCH=$BUILD_PROFILE

# Move compiled binary to standard location
RUN mv stockfish /usr/local/bin/stockfish

# Build Stockfish 8 for concept extraction
WORKDIR /chess
RUN git clone https://github.com/official-stockfish/Stockfish.git stockfish-8 \
    && cd stockfish-8 \
    && git checkout sf_$VERSION_8

WORKDIR /chess/stockfish-8/src
RUN make build ARCH=x86-64

# Move Stockfish 8 binary to standard location
RUN mv stockfish /usr/local/bin/stockfish-8

# --- Stage 2: Install Python Dependencies (changes more frequently) ---
FROM python:3.13-slim-bookworm AS python-deps

ENV PYTHONUNBUFFERED=True
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project definition files for caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# --- Stage 3: Final Application Image ---
FROM python:3.13-slim-bookworm AS runner

WORKDIR /app

# Copy the virtual environment from python-deps stage
COPY --from=python-deps /app/.venv ./.venv

# Copy Stockfish binaries from stockfish-builder stage
COPY --from=stockfish-builder /usr/local/bin/stockfish /usr/local/bin/stockfish
COPY --from=stockfish-builder /usr/local/bin/stockfish-8 /usr/local/bin/stockfish-8

# Copy application code
COPY . /app

# Set environment variables for Stockfish paths
ENV STOCKFISH_PATH=/usr/local/bin/stockfish
ENV STOCKFISH_8_PATH=/usr/local/bin/stockfish-8

# Set the entrypoint
CMD ["/app/.venv/bin/python", "/app/chess_sandbox/server.py"]
