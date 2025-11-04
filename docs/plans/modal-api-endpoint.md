# Modal API Endpoint for Chess Analysis

## Overview

**Goal:** Expose chess engine analysis via a serverless API using Modal, enabling on-demand Stockfish position evaluation through HTTP requests.

**Scope:**
- Single GET endpoint exposing the existing `main()` CLI function
- Stockfish engine only (Maia/Lc0 deferred to future iteration)
- Simple text output response (no complex models needed)
- Reuses all existing CLI logic for consistency

**Non-Goal:** Replace existing Dockerfile - both deployment options serve different use cases.

## Architecture

### Dual Deployment Strategy

| Deployment | Use Case | Advantages |
|------------|----------|------------|
| **Modal (Serverless)** | Production web API | Zero-ops, auto-scaling, pay-per-use, managed infrastructure |
| **Dockerfile (Container)** | Local development, CICD tests | Portable, full control, no vendor lock-in, works offline |

### Why Keep Both?

- **Modal**: Optimized for serverless API workloads with automatic scaling and minimal operational overhead
- **Docker**: Provides flexibility for non-Modal deployments (AWS ECS, GCP Cloud Run, local and cicd testing)

## Key Documentation

- [Modal Image Building](https://modal.com/docs/reference/modal.Image) - Creating container images with system and Python dependencies
- [Modal FastAPI Integration](https://modal.com/docs/guide/webhooks) - Implementing web endpoints with Pydantic validation
- [Modal Continuous Deployment](https://modal.com/docs/guide/continuous-deployment) - GitHub Actions workflow setup

## Implementation Steps

### Phase 1: Modal Endpoint Implementation

**File:** `chess_sandbox/endpoints.py`

**1.1 Build Modal Image**
```python
image = (
    modal.Image.debian_slim()
    .apt_install("stockfish")  # System package from Debian repo
    .pip_install("python-chess", "pydantic", "fastapi[standard]")
    .add_local_python_source("chess_sandbox")  # Include our module
)
```

**Key Details:**
- Stockfish installed via `apt` (avoids manual compilation)
- Binary location: `stockfish` (Debian package default)
- Environment variable: `STOCKFISH_PATH=stockfish`
- Use `.uv_pip_install()` alternative for faster builds (optional optimization)

**1.2 Implement GET Endpoint**
```python
from chess_sandbox.engine.analysis import main

@app.function()
@modal.fastapi_endpoint(method="GET")
def analyze(fen: str, depth: int = 20, num_lines: int = 5) -> str:
    """Analyze a chess position using Stockfish."""
    return main(fen=fen, depth=depth, num_lines=num_lines)
```

**Error Handling:**
- Invalid FEN → Handled by main(), raises ValueError
- Engine failures → Handled by main(), raises RuntimeError
- No additional validation needed - reuses CLI logic

### Phase 2: CI/CD Integration

**File:** `.github/workflows/release.yml`

**2.1 Add Modal Deployment Job**
```yaml
deploy-modal:
  needs: publish
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.13"
    - name: Install Modal
      run: pip install modal
    - name: Deploy to Modal
      env:
        MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
        MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
      run: modal deploy chess_sandbox/endpoints.py
```

**2.2 Configure GitHub Secrets**
- Navigate to repository Settings → Secrets → Actions
- Create `MODAL_TOKEN_ID` from https://modal.com/settings/tokens
- Create `MODAL_TOKEN_SECRET` from same source
- Tokens enable GitHub Actions to authenticate with Modal

### Phase 3: Documentation Updates

**3.1 README.md Additions**
- New section: "Modal API Deployment"
- Prerequisites (Modal account, token setup)
- Local testing: `modal serve chess_sandbox/endpoints.py`
- Production deploy: `modal deploy chess_sandbox/endpoints.py`
- Usage examples with curl/httpie
- Link to this plan document

**3.2 Usage Examples**
```bash
# Start position analysis
curl "https://your-app.modal.run/analyze?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201&depth=20&num_lines=5"

# Custom depth
curl "https://your-app.modal.run/analyze?fen=<FEN>&depth=25&num_lines=3"
```

## API Design

### Request

**Method:** GET
**Path:** `/analyze`
**Query Parameters:**
- `fen` (required): Position in FEN notation
- `depth` (optional, default=20): Stockfish analysis depth
- `num_lines` (optional, default=5): Number of principal variations

### Response

Plain text format (same as CLI output):

```
POSITION:
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Turn: White to move

PRINCIPAL VARIATIONS:

Line 1: Eval +0.17
  Moves: e4 e5 Nf3 Nc6

Line 2: Eval +0.15
  Moves: d4 Nf6 c4 e6
...
```

**Status Codes:**
- `200 OK`: Successful analysis (text response)
- `400 Bad Request`: Invalid FEN
- `500 Internal Server Error`: Engine failure

## Testing Strategy

### Local Testing (Development)
```bash
# Start Modal dev server
modal serve chess_sandbox/endpoints.py

# Test with curl
curl "http://localhost:8000/analyze?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201"
```

### Test Cases
1. **Valid FEN**: Verify text output with position and variations
2. **Invalid FEN**: Confirm 400 error with helpful message
3. **Edge cases**: Empty board, checkmate positions
4. **Parameter validation**: Default depth/num_lines values
5. **Consistency**: Verify output matches CLI `main()` function

### Production Deployment
```bash
# Deploy to Modal cloud
modal deploy chess_sandbox/endpoints.py

# Test production endpoint
curl "https://your-app.modal.run/analyze?fen=<FEN>"
```

## Deployment Instructions

### Prerequisites
1. **Modal Account**: Sign up at https://modal.com
2. **Create Token**: https://modal.com/settings/tokens
3. **Install Modal CLI**: `pip install modal`
4. **Authenticate**: `modal token set --token-id <ID> --token-secret <SECRET>`

### Local Development
```bash
# Install dependencies
uv sync

# Start dev server
modal serve chess_sandbox/endpoints.py

# Access at http://localhost:8000
```

### Production Deployment
```bash
# Manual deployment
modal deploy chess_sandbox/endpoints.py

# Automatic deployment (GitHub releases)
# Triggers on release publish via .github/workflows/release.yml
```

## Future Enhancements

1. **Maia/Lc0 Support**: Add human-like move ranking
   - Requires weight files in Modal image (~200MB)
   - Separate endpoint or query parameter flag
2. **POST Endpoint**: Accept multiple positions for batch analysis
3. **WebSocket Endpoint**: Streaming analysis updates
4. **Rate Limiting**: Protect against abuse
5. **Caching**: Cache common positions (e.g., opening positions)
6. **Metrics**: Track usage, latency, error rates

## References

- Existing CLI: `chess_sandbox/engine_analysis.py:main()`
- Docker implementation: `Dockerfile` (multi-stage build with compiled Stockfish)
- Analysis logic: `chess_sandbox/engine_analysis.py:analyze_variations()`
- Pydantic models: `chess_sandbox/engine_analysis.py:PrincipalVariation`, `CandidateMove`
