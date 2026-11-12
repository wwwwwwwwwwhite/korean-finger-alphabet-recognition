#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.9+ is required. Install Python from python.org or Homebrew." >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node.js 18+ and npm are required." >&2
  exit 1
fi
if ! /usr/bin/env python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "Python 3.9 or later is required." >&2
  exit 1
fi
if ! node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)'; then
  echo "Node.js 18 or later is required." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt

if [[ ! -d node_modules ]]; then
  npm install
fi

backend_pid=""
frontend_pid=""
cleanup() {
  if [[ -n "$frontend_pid" ]]; then kill "$frontend_pid" 2>/dev/null || true; fi
  if [[ -n "$backend_pid" ]]; then kill "$backend_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
backend_pid=$!

backend_ready=0
for ((attempt = 0; attempt < 30; attempt += 1)); do
  if curl --fail --silent http://127.0.0.1:8000/api/health >/dev/null; then
    backend_ready=1
    break
  fi
  sleep 1
done
if [[ "$backend_ready" -ne 1 ]]; then
  echo "The API did not become ready at http://127.0.0.1:8000." >&2
  exit 1
fi

npm run dev -- --host 127.0.0.1 &
frontend_pid=$!

sleep 3
open http://localhost:5173
echo "Son is running at http://localhost:5173 (press Ctrl+C to stop)."

wait "$backend_pid" "$frontend_pid"
