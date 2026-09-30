#!/usr/bin/env bash
# Starts the CARLA server (unless one is already listening) and the spectral live viewer.
#
#   scripts/run_carla_live.sh                                  # config.example.json, CARLA window on
#   scripts/run_carla_live.sh --config my_live.json --offscreen
#   scripts/run_carla_live.sh --no-server --host 10.0.0.5      # CARLA running elsewhere
#   scripts/run_carla_live.sh -- --frames 100 --backend cpu    # extra spectral_live.py arguments after --
#
# Needs env.sh (scripts/setup.sh) and CARLA_ROOT (scripts/install_carla.sh or export CARLA_ROOT=...).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG="$ROOT/tools/carla_live/config.example.json"
HOST=localhost
PORT=2000
OFFSCREEN=0
QUALITY=Epic
START_SERVER=1
WAIT=180
EXTRA=()
CARLA_ROOT_ARG=""

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; cat <<'EOF'
Options:
  --config FILE        viewer config (default tools/carla_live/config.example.json)
  --host H --port P    CARLA server address (default localhost:2000)
  --offscreen          start CARLA with -RenderOffScreen (no UE window; the viewer still shows CARLA RGB)
  --quality Low|Epic   CARLA rendering quality (default Epic; Low leaves more GPU time for the renderer)
  --carla-root DIR     CARLA install directory (default $CARLA_ROOT)
  --no-server          do not start CARLA, only connect
  --wait SEC           max seconds to wait for the server (default 180)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$(realpath "$2"); shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --offscreen) OFFSCREEN=1; shift ;;
    --quality) QUALITY=$2; shift 2 ;;
    --carla-root) CARLA_ROOT_ARG=$(realpath "$2"); shift 2 ;;
    --no-server) START_SERVER=0; shift ;;
    --wait) WAIT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option: $1"; usage; exit 2 ;;
  esac
done

[ -f "$ROOT/env.sh" ] || { echo "env.sh missing: run scripts/setup.sh first"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/env.sh"
[ -n "${CARLA_ROOT_ARG:-}" ] && export CARLA_ROOT="$CARLA_ROOT_ARG"   # --carla-root wins over env.local.sh
python -c 'import spectral_renderer' 2>/dev/null || { echo "spectral_renderer module not built: run scripts/setup.sh"; exit 1; }
python -c 'import carla' 2>/dev/null || { echo "carla client missing: scripts/install_carla.sh (or pip install carla==0.9.15)"; exit 1; }

port_open() { python - "$1" "$2" <<'EOF'
import socket, sys
s = socket.socket(); s.settimeout(1.0)
sys.exit(0 if s.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
EOF
}

SERVER_PGID=""
cleanup() {
  if [ -n "$SERVER_PGID" ]; then
    echo "stopping CARLA (process group $SERVER_PGID)"
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if port_open "$HOST" "$PORT"; then
  echo "CARLA already listening on $HOST:$PORT"
elif [ $START_SERVER = 1 ]; then
  [ "$HOST" = localhost ] || [ "$HOST" = 127.0.0.1 ] || { echo "no server at $HOST:$PORT (remote hosts are not started)"; exit 1; }
  [ -n "${CARLA_ROOT:-}" ] && [ -x "$CARLA_ROOT/CarlaUE4.sh" ] || {
    echo "CARLA_ROOT not set or CarlaUE4.sh missing: scripts/install_carla.sh, or --carla-root DIR"; exit 1; }
  ARGS=(-carla-rpc-port="$PORT" -quality-level="$QUALITY")
  [ $OFFSCREEN = 1 ] && ARGS+=(-RenderOffScreen)
  LOG="$ROOT/carla_server.log"
  echo "starting $CARLA_ROOT/CarlaUE4.sh ${ARGS[*]}  (log: $LOG)"
  setsid "$CARLA_ROOT/CarlaUE4.sh" "${ARGS[@]}" > "$LOG" 2>&1 &
  SERVER_PGID=$!
  for ((i = 0; i < WAIT; i++)); do
    port_open "$HOST" "$PORT" && break
    kill -0 "$SERVER_PGID" 2>/dev/null || { echo "CARLA exited early; see $LOG"; tail -20 "$LOG"; SERVER_PGID=""; exit 1; }
    sleep 1
  done
  port_open "$HOST" "$PORT" || { echo "CARLA did not open port $PORT within ${WAIT}s; see $LOG"; exit 1; }
  echo "CARLA is up after ${i}s; waiting for the world to load"
  sleep 5
else
  echo "no CARLA server at $HOST:$PORT (started without --no-server to launch one)"; exit 1
fi

cd "$ROOT"
python tools/carla_live/spectral_live.py --config "$CONFIG" --host "$HOST" --port "$PORT" "${EXTRA[@]}"
