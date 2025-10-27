# --- Stage 1: Build Stockfish (rarely changes, cached longer) ---
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

# Copy Stockfish binary from stockfish-builder stage
COPY --from=stockfish-builder /usr/local/bin/stockfish /usr/local/bin/stockfish

# Copy application code
COPY . /app

# Set environment variable for Stockfish path
ENV STOCKFISH_PATH=/usr/local/bin/stockfish

# Set the entrypoint
CMD ["/app/.venv/bin/python", "/app/chess_sandbox/server.py"]
