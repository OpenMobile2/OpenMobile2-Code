#!/usr/bin/env bash
# Mac counterpart of run_eval_session.ps1 (bash; no PowerShell required).
#
#   cd <OPENMOBILE_ROOT>/MobileGym
#   ./run_eval_session_mac.sh -ModelName YOUR_MODEL
#
# Results: runs/<ModelName dots removed>
# Re-run same ModelName -> resume_eval_mg.py (merge into same folder).
#
# Explicit resume:
#   ./run_eval_session_mac.sh -Resume runs/YOUR_MODEL
#
# L1 smoke:
#   ./run_eval_session_mac.sh -Split "" -FilterDifficulty L1
#
# Thought: training format (see ../qwen35_thought_session.md):
#   ./run_eval_session_mac.sh -Runtime qwen35_thought_session -LastN 3

set -euo pipefail

WorkDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WorkDir"

FrontendDir=""
EnvUrl=""
Port=4172
Parallel=8
Processes=1
Split="test"
TaskId=""
Suite=""
FilterDifficulty=""
Runtime="qwen35_thought_session"
LastN=3
EnableThinking="true"
RequireThinkTags="false"
Qwen35ToolCallMode="native"
ModelBaseUrl="http://<openai-compatible-host>/v1"
ModelName="YOUR_MODEL"
ApiKey="EMPTY"
NodeCondaEnv="mobilegym"
PythonCondaEnv="android_world"
PythonExe=""
HealthTimeoutMinutes=8
RunsDir="runs"
Resume=""

preview_pid=""
caffeinate_pid=""
started_preview=0
exit_code=1

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "error: $*"
  exit 1
}

usage() {
  sed -n '2,16p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    -FrontendDir|--frontend-dir) FrontendDir="$2"; shift 2 ;;
    -EnvUrl|--env-url) EnvUrl="$2"; shift 2 ;;
    -Port|--port) Port="$2"; shift 2 ;;
    -Parallel|--parallel) Parallel="$2"; shift 2 ;;
    -Processes|--processes) Processes="$2"; shift 2 ;;
    -Split|--split) Split="$2"; shift 2 ;;
    -TaskId|--task-id) TaskId="$2"; shift 2 ;;
    -Suite|--suite) Suite="$2"; shift 2 ;;
    -FilterDifficulty|--filter-difficulty) FilterDifficulty="$2"; shift 2 ;;
    -Runtime|--runtime) Runtime="$2"; shift 2 ;;
    -LastN|--last-n) LastN="$2"; LastNSet=1; shift 2 ;;
    -EnableThinking|--enable-thinking) EnableThinking="$2"; shift 2 ;;
    -RequireThinkTags|--require-think-tags) RequireThinkTags="$2"; shift 2 ;;
    -Qwen35ToolCallMode|--qwen35-tool-call-mode) Qwen35ToolCallMode="$2"; shift 2 ;;
    -ModelBaseUrl|--model-base-url) ModelBaseUrl="$2"; shift 2 ;;
    -ModelName|--model-name) ModelName="$2"; shift 2 ;;
    -ApiKey|--api-key) ApiKey="$2"; shift 2 ;;
    -NodeCondaEnv|--node-conda-env) NodeCondaEnv="$2"; shift 2 ;;
    -PythonCondaEnv|--python-conda-env) PythonCondaEnv="$2"; shift 2 ;;
    -PythonExe|--python-exe) PythonExe="$2"; shift 2 ;;
    -HealthTimeoutMinutes|--health-timeout-minutes) HealthTimeoutMinutes="$2"; shift 2 ;;
    -RunsDir|--runs-dir) RunsDir="$2"; RunsDirSet=1; shift 2 ;;
    -Resume|--resume) Resume="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

LastNSet="${LastNSet:-0}"
RunsDirSet="${RunsDirSet:-0}"
if [[ "$LastNSet" -eq 0 && ( "$Runtime" == "venus" || "$Runtime" == "qwen3vl" ) ]]; then
  LastN=1
fi
if [[ "$RunsDirSet" -eq 0 && "$Runtime" == "qwen3vl" ]]; then
  RunsDir="${RunsDir}/qwen3vl"
fi
if [[ "$Runtime" == "qwen35_thought_session" ]]; then
  EnableThinking="false"
  Qwen35ToolCallMode="xml"
fi

if [[ -z "$FrontendDir" ]]; then
  FrontendDir="$(cd "$WorkDir/../../mobilegym/mobilegym" && pwd)"
fi
if [[ -z "$EnvUrl" ]]; then
  EnvUrl="http://127.0.0.1:${Port}"
fi

conda_base() {
  local home="${HOME:-}"
  local candidate
  for candidate in \
    "$home/miniconda3" \
    "$home/anaconda3" \
    "$home/miniforge3" \
    "$home/mambaforge"
  do
    if [[ -d "$candidate/envs" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    candidate="$(conda info --base 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    if [[ -n "$candidate" && -d "$candidate/envs" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  die "conda base not found. Expected ~/miniconda3 or conda info --base."
}

conda_env_bin() {
  local env_name="$1" name="$2"
  local path
  path="$(conda_base)/envs/${env_name}/bin/${name}"
  [[ -x "$path" ]] || die "$name not found in conda env '$env_name'. Expected: $path"
  printf '%s\n' "$path"
}

eval_python() {
  if [[ -n "$PythonExe" && -x "$PythonExe" ]]; then
    printf '%s\n' "$PythonExe"
    return 0
  fi
  conda_env_bin "$PythonCondaEnv" python
}

frontend_up() {
  local url="$1"
  curl -fsS --max-time 5 "$url" >/dev/null 2>&1
}

enable_keep_awake() {
  if [[ -x /usr/bin/caffeinate ]]; then
    /usr/bin/caffeinate -dims &
    caffeinate_pid=$!
    log "Keep-awake on (caffeinate)."
  else
    log "caffeinate not found; skip keep-awake."
  fi
}

stop_tree() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  pkill -P "$pid" >/dev/null 2>&1 || true
  kill -TERM "$pid" >/dev/null 2>&1 || true
  sleep 0.4
  kill -KILL "$pid" >/dev/null 2>&1 || true
}

cleanup() {
  if [[ "$started_preview" -eq 1 && -n "$preview_pid" ]]; then
    log "Stopping frontend preview ..."
    stop_tree "$preview_pid"
  fi
  if [[ -n "$caffeinate_pid" ]]; then
    stop_tree "$caffeinate_pid"
  fi
}

trap cleanup EXIT

safe_run_name() {
  local safe="${1//./}"
  safe="$(printf '%s' "$safe" | tr -s '[:space:]/\\:*?"<>|' '_')"
  safe="$(printf '%s' "$safe" | sed -E 's/_+/_/g; s/^[._]+//; s/[._]+$//')"
  [[ -n "$safe" ]] || safe="run"
  printf '%s\n' "$safe"
}

start_frontend() {
  local dir="$1" listen_port="$2" url="$3"
  [[ -d "$dir" ]] || die "Frontend dir not found: $dir"
  if frontend_up "$url"; then
    log "Frontend already up at ${url}, skip deploy."
    return 0
  fi

  local npm
  npm="$(conda_env_bin "$NodeCondaEnv" npm)"
  (
    cd "$dir"
    if [[ ! -d node_modules ]]; then
      log "npm install ..."
      "$npm" install
    fi
    if [[ ! -f dist/index.html ]]; then
      log "npm run build ..."
      "$npm" run build
    fi
  )
  log "Starting preview on port ${listen_port} ..."
  (
    cd "$dir"
    exec "$npm" run preview -- --host 127.0.0.1 --port "$listen_port"
  ) &
  preview_pid=$!
  started_preview=1

  local deadline=$((SECONDS + HealthTimeoutMinutes * 60))
  while (( SECONDS < deadline )); do
    if frontend_up "$url"; then
      log "Frontend healthy: ${url}"
      return 0
    fi
    if ! kill -0 "$preview_pid" >/dev/null 2>&1; then
      die "npm preview exited early."
    fi
    log "Waiting for frontend ..."
    sleep 3
  done
  die "Frontend not reachable at $url within ${HealthTimeoutMinutes} minutes."
}

enable_keep_awake
evalPython="$(eval_python)"
export MOBILEGYM_FRONTEND="$FrontendDir"

log "npm from conda env ${NodeCondaEnv}; python from conda env ${PythonCondaEnv}"
log "python=${evalPython}"
if ! "$evalPython" -c "import requests, playwright, openai, PIL"; then
  die "android_world env is missing packages. Run: conda activate android_world; pip install requests -r ${FrontendDir}/bench_env/requirements.txt; playwright install chromium"
fi

split_label="ALL"
if [[ -n "$TaskId" ]]; then
  split_label="task:${TaskId}"
elif [[ -n "$Suite" ]]; then
  split_label="suite:${Suite}"
elif [[ -n "$Split" ]]; then
  split_label="$Split"
elif [[ -n "$FilterDifficulty" ]]; then
  split_label="difficulty:${FilterDifficulty}"
fi
log "MobileGym EVAL  split=${split_label} agent=${Runtime} last_n=${LastN} thinking=${EnableThinking} require_think_tags=${RequireThinkTags} mode=${Qwen35ToolCallMode}"
log "frontend=${FrontendDir} env=${EnvUrl} parallel=${Parallel} processes=${Processes}"

start_frontend "$FrontendDir" "$Port" "$EnvUrl"

if [[ "$RunsDir" = /* ]]; then
  runsRoot="$RunsDir"
else
  runsRoot="$(cd "$WorkDir" && mkdir -p "$RunsDir" && cd "$RunsDir" && pwd)"
fi
fixedRunDir="${runsRoot}/$(safe_run_name "$ModelName")"

common=(
  --env-url "$EnvUrl"
  --agent "$Runtime"
  --parallel "$Parallel"
  --processes "$Processes"
  --headless
  --last_n "$LastN"
  --enable_thinking "$EnableThinking"
  --require_think_tags "$RequireThinkTags"
  --qwen35_tool_call_mode "$Qwen35ToolCallMode"
  --model-base-url "$ModelBaseUrl"
  --model-name "$ModelName"
  --model-api-key "$ApiKey"
  --runs-dir "$runsRoot"
)
if [[ -n "$TaskId" ]]; then
  common+=(--task-id "$TaskId")
elif [[ -n "$Suite" ]]; then
  common+=(--suite "$Suite")
elif [[ -n "$Split" ]]; then
  common+=(--split "$Split")
elif [[ -n "$FilterDifficulty" ]]; then
  common+=(--filter-difficulty "$FilterDifficulty")
fi

target=""
if [[ -n "$Resume" ]]; then
  if [[ "$Resume" = /* ]]; then
    target="$Resume"
  else
    target="$(cd "$WorkDir" && cd "$(dirname "$Resume")" && pwd)/$(basename "$Resume")"
  fi
elif [[ -f "${fixedRunDir}/meta.json" ]]; then
  target="$fixedRunDir"
fi

printf '%s\n' "$(printf '=%.0s' {1..72})"
set +e
if [[ -n "$target" ]]; then
  log "OpenMobile resume (merge into): ${target}"
  log "python resume_eval_mg.py --run-dir ${target} ${common[*]}"
  "$evalPython" resume_eval_mg.py --run-dir "$target" "${common[@]}"
  exit_code=$?
else
  log "Fresh run dir: ${fixedRunDir}"
  log "python run_eval_mg.py ${common[*]} --run-dir ${fixedRunDir}"
  "$evalPython" run_eval_mg.py "${common[@]}" --run-dir "$fixedRunDir"
  exit_code=$?
fi
set -e
printf '%s\n' "$(printf '=%.0s' {1..72})"
log "eval exited with code ${exit_code}"
exit "$exit_code"
