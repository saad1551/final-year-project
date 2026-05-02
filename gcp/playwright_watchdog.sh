#!/usr/bin/env bash
# Playwright server watchdog.
#
# Runs in its own screen session alongside the training tmux. Every 20s,
# checks that port 3000 is actually accepting connections. If not:
#   1. SIGSTOP the python training process so it doesn't burn trajectories
#      while the server is down (each /start failure costs ~30s of clock).
#   2. Kill any stuck node/screen leftovers, restart start_playwright_server.sh.
#   3. Wait for the port to come back up.
#   4. SIGCONT the python process so it resumes from where it paused.
#
# Logs every event to ~/playwright_watchdog.log with timestamps.

set -u

LOG="${LOG:-$HOME/playwright_watchdog.log}"
PORT="${PORT:-3000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
RESTART_GRACE="${RESTART_GRACE:-12}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

is_listening() {
  ss -ltn 2>/dev/null | grep -q ":${PORT} "
}

get_python_pid() {
  pgrep -f "pipeline_in_steps.py" | head -1
}

restart_playwright() {
  log "  killing leftover server bits"
  pkill -9 -f "node.*server/src/index.js" 2>/dev/null || true
  pkill -9 -f "xargs.*server/src/index.js" 2>/dev/null || true
  screen -S playwright -X quit 2>/dev/null || true
  sleep 3
  log "  launching start_playwright_server.sh"
  cd "$HOME/final-year-project" && bash start_playwright_server.sh
  sleep "$RESTART_GRACE"
}

log "watchdog started (port=$PORT, interval=${CHECK_INTERVAL}s)"

while true; do
  if ! is_listening; then
    log "port $PORT is DOWN"
    PYPID=$(get_python_pid || echo "")
    if [[ -n "$PYPID" ]]; then
      log "  pausing python pid=$PYPID"
      kill -STOP "$PYPID" 2>/dev/null || true
    fi
    restart_playwright
    if is_listening; then
      log "  server back up"
    else
      log "  RESTART FAILED — server still down after grace"
    fi
    if [[ -n "$PYPID" ]] && kill -0 "$PYPID" 2>/dev/null; then
      log "  resuming python pid=$PYPID"
      kill -CONT "$PYPID" 2>/dev/null || true
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
