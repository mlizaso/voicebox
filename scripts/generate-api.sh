#!/bin/bash
# Generate OpenAPI TypeScript client from FastAPI schema

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STARTED_BACKEND=false
BACKEND_PID=""

cleanup() {
    if [ "$STARTED_BACKEND" = true ] && [ -n "$BACKEND_PID" ]; then
        echo "Stopping backend server..."
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
        STARTED_BACKEND=false
    fi
}

trap cleanup EXIT

echo "Generating OpenAPI client..."

# Check if backend is running
if ! curl -s http://localhost:17493/openapi.json > /dev/null 2>&1; then
    echo "Backend not running. Starting backend..."

    # Check if virtual environment exists
    if [ ! -d "$REPO_ROOT/backend/venv" ]; then
        echo "Creating virtual environment..."
        python -m venv "$REPO_ROOT/backend/venv"
    fi

    # The generated environment is intentionally absent from source control.
    # shellcheck disable=SC1091
    source "$REPO_ROOT/backend/venv/bin/activate" 2>/dev/null || source "$REPO_ROOT/backend/venv/Scripts/activate" 2>/dev/null

    # Install dependencies if needed
    if ! python -c "import fastapi" 2>/dev/null; then
        echo "Installing backend dependencies..."
        python -m pip install -r "$REPO_ROOT/backend/requirements.txt"
    fi

    # Start backend in background
    echo "Starting backend server..."
    (
        cd "$REPO_ROOT"
        exec uvicorn backend.main:app --port 17493
    ) &
    BACKEND_PID=$!
    STARTED_BACKEND=true

    # Wait for server to be ready
    echo "Waiting for server to start..."
    for _ in {1..30}; do
        if curl -s http://localhost:17493/openapi.json > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:17493/openapi.json > /dev/null 2>&1; then
        echo "Error: Backend failed to start"
        exit 1
    fi

    echo "Backend started (PID: $BACKEND_PID)"
fi

# Download OpenAPI schema
echo "Downloading OpenAPI schema..."
curl -fsS http://localhost:17493/openapi.json > "$REPO_ROOT/app/openapi.json"

# Check if openapi-typescript-codegen is installed
if ! (cd "$REPO_ROOT" && bunx --bun openapi-typescript-codegen --version > /dev/null 2>&1); then
    echo "Installing openapi-typescript-codegen..."
    (cd "$REPO_ROOT" && bun add -d openapi-typescript-codegen)
fi

# Generate TypeScript client
echo "Generating TypeScript client..."
(
    cd "$REPO_ROOT/app"
    bunx --bun openapi-typescript-codegen \
        --input openapi.json \
        --output src/lib/api \
        --client fetch \
        --useOptions \
        --exportSchemas true
)

# The generator owns core/request.ts. Re-apply the shared authenticated
# transport so remote bearer/session support cannot disappear on regeneration.
python "$REPO_ROOT/scripts/patch-generated-api-auth.py" "$REPO_ROOT/app/src/lib/api/core/request.ts"

echo "API client generated in app/src/lib/api"

cleanup
trap - EXIT

echo "Done!"
