# --- Stage 1: Install Python Dependencies ---
# Platform note: CI/CD and Modal use linux/amd64. Local dev on Apple Silicon uses linux/arm64.
# Both platforms supported via torchvision 0.23.0 (see pyproject.toml for compatibility notes).
# To force a specific platform: docker build --platform=linux/amd64 ...
FROM python:3.13-slim-bookworm AS python-deps

ENV PYTHONUNBUFFERED=True
WORKDIR /app

# Install uv (from linux/amd64 variant)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project definition files for caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# --- Stage 2: Final Application Image ---
FROM python:3.13-slim-bookworm AS runner

WORKDIR /app

# Install chess engines and wget for downloading weights
# - stockfish: Available in Debian repos (v17)
# - lc0: Download pre-compiled binary from unofficial builds
#   Source: https://github.com/jldblog/lc0-linux-unofficial-builds
RUN apt-get update \
    && apt-get install -y wget stockfish \
    && wget -q -O /usr/local/bin/lc0 \
       https://raw.githubusercontent.com/jldblog/lc0-linux-unofficial-builds/main/v0.31/lc0-v0.31-linux-cpu \
    && chmod +x /usr/local/bin/lc0 \
    && mkdir -p /app/data \
    && wget -q -O /app/data/maia-1100.pb.gz \
       https://raw.githubusercontent.com/CSSLab/maia-chess/master/maia_weights/maia-1100.pb.gz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from python-deps stage
COPY --from=python-deps /app/.venv ./.venv

# Copy application code
COPY . /app

# Set environment variables for engine paths
# Note: Debian apt installs stockfish to /usr/games/
ENV STOCKFISH_PATH=/usr/games/stockfish
ENV LC0_PATH=/usr/local/bin/lc0
ENV MAIA_WEIGHTS_PATH=/app/data/maia-1100.pb.gz

# Set the entrypoint
CMD ["/app/.venv/bin/python", "/app/chess_sandbox/server.py"]
