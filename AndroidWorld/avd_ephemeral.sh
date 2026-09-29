#!/usr/bin/env bash
#
# avd_ephemeral.sh — quickly make & destroy throw-away AVD clones.
#
# An "ephemeral" clone is an APFS clonefile (`cp -c`, copy-on-write) of a source
# <AVD>.avd directory. It is instant and costs ~0 disk until written to, so you can
# spin up a disposable phone, do whatever you want (random walk, app setup experiments,
# manual poking, run_diy, ...), then throw it away WITHOUT ever touching your real AVD.
#
# Clones are named "<src>-ephem-<ts>-<pid>" so `gc` / `list` can find them.
#
# Subcommands
# -----------
#   clone   <src> [name]                 Clone <src>; print the clone's AVD name. Nothing is booted.
#   boot    <avd>  [-p PORT] [-g GRPC]    Boot an AVD *detached* (survives this shell); print port/pid.
#                  [--headless 0|1] [--wait]
#   destroy <avd>                         Kill it if running, then rm -rf the clone dir + .ini.
#   gc                                    Destroy ALL not-running "*-ephem-*" clones.
#   list                                  List ephemeral clones (and whether each is running).
#   run     <src> [-p PORT] [-g GRPC]     Clone+boot+run a command with $CONSOLE_PORT/$GRPC_PORT
#                  [--headless 0|1] -- <cmd...>   exported, then auto-destroy on exit (any outcome).
#
# Examples
# --------
#   # Disposable phone you drive yourself, then clean up:
#   name=$(./avd_ephemeral.sh clone MineAndroid-general-user)
#   ./avd_ephemeral.sh boot "$name" --wait        # prints CONSOLE_PORT / GRPC_PORT
#   python tooling/setup_emulator.py --console_port=... --grpc_port=...
#   ./avd_ephemeral.sh destroy "$name"
#
#   # One-shot managed lifecycle (clone -> boot -> run -> destroy):
#   ./avd_ephemeral.sh run MineAndroid-general-user -- \
#       python run_diy.py --console_port=$CONSOLE_PORT --grpc_port=$GRPC_PORT ...
#
#   # Housekeeping:
#   ./avd_ephemeral.sh list
#   ./avd_ephemeral.sh gc
#

# 两种用法

# A. 手动/半自动（配任意工具，自己掌控生命周期）：
# cd AndroidWorld
# name=$(./avd_ephemeral.sh clone MineAndroid-general-user)   # 秒级
# eval "$(./avd_ephemeral.sh boot "$name" --wait)"            # 导出 CONSOLE_PORT/GRPC_PORT
# python tooling/setup_emulator.py --console_port=$CONSOLE_PORT --grpc_port=$GRPC_PORT
# # ...爱干嘛干嘛，源 AVD 全程不动...
# ./avd_ephemeral.sh destroy "$name"

# B. 一条命令全托管（命令里用 {{CONSOLE_PORT}}/{{GRPC_PORT}} 占位符，或读同名 env）：
# ./avd_ephemeral.sh run MineAndroid-general-user -- \
#     python run_diy.py --console_port={{CONSOLE_PORT}} --grpc_port={{GRPC_PORT}} --task=...

# run_random_walk_ephemeral.sh 现在只是薄封装，转调 avd_ephemeral.sh run（单一真源）。
# set -euo pipefail

# --------------------------------------------------------------------------- sdk paths
AVD_HOME="${ANDROID_AVD_HOME:-$HOME/.android/avd}"
_sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
EMULATOR="${EMULATOR_BIN:-$_sdk/emulator/emulator}"
ADB="${ADB_BIN:-$_sdk/platform-tools/adb}"
EPHEM_TAG="-ephem-"   # naming marker for list/gc

_need_tools() {
  [[ -x "$EMULATOR" ]] || { echo "ERROR: emulator not found at $EMULATOR (set ANDROID_HOME or EMULATOR_BIN)" >&2; exit 1; }
  [[ -x "$ADB" ]]      || { echo "ERROR: adb not found at $ADB (set ANDROID_HOME or ADB_BIN)" >&2; exit 1; }
}

# --------------------------------------------------------------------------- helpers
_is_running() {  # $1 = avd name ; true if an emulator process has "-avd <name> "
  pgrep -fl "\-avd ${1} " >/dev/null 2>&1
}

_port_in_use() {  # $1 = console port
  "$ADB" devices 2>/dev/null | grep -q "emulator-$1[[:space:]]" && return 0
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1 && return 0
  fi
  return 1
}

_pick_ports() {  # echoes "CONSOLE GRPC"; honors $1/$2 if given
  local console="${1:-}" grpc="${2:-}"
  if [[ -z "$console" ]]; then
    local p=5554
    while _port_in_use "$p" || _port_in_use "$((p+1))"; do
      p=$((p+2)); [[ $p -gt 5680 ]] && { echo "ERROR: no free console port in 5554..5680" >&2; exit 1; }
    done
    console="$p"
  fi
  [[ -z "$grpc" ]] && grpc="$((console + 3000))"
  echo "$console $grpc"
}

# Clone <src> into <clone>; patch every place the name is baked in. Prints nothing.
_do_clone() {  # $1 = src name, $2 = clone name
  local src="$1" clone="$2"
  local src_dir="$AVD_HOME/$src.avd" clone_dir="$AVD_HOME/$clone.avd" clone_ini="$AVD_HOME/$clone.ini"
  [[ -d "$src_dir" ]] || { echo "ERROR: source AVD dir not found: $src_dir" >&2; exit 1; }
  if _is_running "$src"; then
    echo "ERROR: source AVD '$src' is currently running. Stop it before cloning (its on-disk state is stale)." >&2
    exit 1
  fi
  [[ -e "$clone_dir" ]] && { echo "ERROR: clone target already exists: $clone_dir" >&2; exit 1; }

  # -c = APFS clonefile (copy-on-write, instant). Falls back to a full copy off-APFS.
  cp -c -R "$src_dir" "$clone_dir" 2>/dev/null || cp -R "$src_dir" "$clone_dir"
  rm -f "$clone_dir"/*.lock "$clone_dir"/multiinstance.lock 2>/dev/null || true

  cat > "$clone_ini" <<EOF
avd.ini.encoding=UTF-8
path=$clone_dir
path.rel=avd/$clone.avd
target=android-33
EOF

  # Rewrite the source name -> clone name inside the config files that bake it in.
  # Fixes disk paths AND avd.name/avd.id, so a quickboot from default_boot still matches.
  local patch=("$clone_dir/config.ini" "$clone_dir/hardware-qemu.ini" "$clone_dir/emu-launch-params.txt")
  local hw
  while IFS= read -r hw; do patch+=("$hw"); done < <(find "$clone_dir/snapshots" -name hardware.ini 2>/dev/null)
  local f
  for f in "${patch[@]}"; do
    [[ -f "$f" ]] && sed -i '' "s#${src}#${clone}#g" "$f"
  done
}

_gen_clone_name() {  # $1 = src
  echo "${1}${EPHEM_TAG}$(date +%s)-$$"
}

_do_destroy() {  # $1 = avd name
  local avd="$1"
  local dir="$AVD_HOME/$avd.avd" ini="$AVD_HOME/$avd.ini"
  if _is_running "$avd"; then
    # find its console port from `adb devices`? Kill via any emulator-<port> that matches.
    local pid port
    pid=$(pgrep -f "\-avd ${avd} " | head -1)
    port=$(pgrep -af "\-avd ${avd} " 2>/dev/null | grep -oE '\-port [0-9]+' | grep -oE '[0-9]+' | head -1)
    echo ">>> [destroy] killing $avd (port ${port:-?}, pid ${pid:-?})"
    [[ -n "$port" ]] && "$ADB" -s "emulator-$port" emu kill >/dev/null 2>&1 || true
    if [[ -n "$pid" ]]; then
      for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -rf "$dir" "$ini"
  echo ">>> [destroy] removed $avd"
}

# Boot <avd> detached. Echoes "CONSOLE GRPC PID". Optional wait for boot_completed.
_do_boot() {  # $1 avd, $2 console, $3 grpc, $4 headless(0/1), $5 wait(0/1), $6 boot_timeout
  local avd="$1" console="$2" grpc="$3" headless="$4" wait="$5" timeout="${6:-300}"
  local dir="$AVD_HOME/$avd.avd" log="$AVD_HOME/$avd.emulog"
  [[ -d "$dir" ]] || { echo "ERROR: AVD not found: $dir" >&2; exit 1; }

  local args=(-avd "$avd" -port "$console" -grpc "$grpc" -no-audio -no-snapshot-save)
  [[ "$headless" == "1" ]] && args+=(-no-window)
  [[ -d "$dir/snapshots/default_boot" ]] && args+=(-snapshot default_boot)
  # shellcheck disable=SC2206
  [[ -n "${EMULATOR_EXTRA_ARGS:-}" ]] && args+=(${EMULATOR_EXTRA_ARGS})

  nohup "$EMULATOR" "${args[@]}" >"$log" 2>&1 &
  local pid=$!
  disown "$pid" 2>/dev/null || true

  if [[ "$wait" == "1" ]]; then
    echo ">>> Waiting for boot (timeout ${timeout}s)... log: $log" >&2
    local deadline=$((SECONDS + timeout))
    "$ADB" -s "emulator-$console" wait-for-device
    until [[ "$("$ADB" -s "emulator-$console" shell getprop sys.boot_completed 2>/dev/null | tr -d '[:space:]')" == "1" ]]; do
      kill -0 "$pid" 2>/dev/null || { echo "ERROR: emulator died during boot. Log tail:" >&2; tail -30 "$log" >&2; exit 1; }
      [[ $SECONDS -ge $deadline ]] && { echo "ERROR: boot timed out. Log tail:" >&2; tail -30 "$log" >&2; exit 1; }
      sleep 2
    done
    echo ">>> Boot complete." >&2
  fi
  echo "$console $grpc $pid"
}

# --------------------------------------------------------------------------- subcommands
cmd_clone() {
  local src="${1:?usage: clone <src> [name]}"; shift || true
  local clone="${1:-$(_gen_clone_name "$src")}"
  _do_clone "$src" "$clone"
  echo ">>> Cloned $src -> $clone" >&2
  echo "$clone"          # stdout = clone name, so you can capture it
}

cmd_boot() {
  local avd="" console="" grpc="" headless="1" wait="0" timeout="300"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--port)  console="$2"; shift 2;;
      -g|--grpc)  grpc="$2"; shift 2;;
      --headless) headless="$2"; shift 2;;
      --wait)     wait="1"; shift;;
      --timeout)  timeout="$2"; shift 2;;
      *) [[ -z "$avd" ]] && { avd="$1"; shift; } || { echo "unexpected arg: $1" >&2; exit 1; };;
    esac
  done
  [[ -n "$avd" ]] || { echo "usage: boot <avd> [-p PORT] [-g GRPC] [--headless 0|1] [--wait]" >&2; exit 1; }
  read -r console grpc < <(_pick_ports "$console" "$grpc")
  local out; out=$(_do_boot "$avd" "$console" "$grpc" "$headless" "$wait" "$timeout")
  read -r console grpc pid <<<"$out"
  echo ">>> Booted $avd  ->  CONSOLE_PORT=$console  GRPC_PORT=$grpc  pid=$pid" >&2
  echo "CONSOLE_PORT=$console GRPC_PORT=$grpc"   # stdout: eval-friendly
}

cmd_destroy() {
  local avd="${1:?usage: destroy <avd>}"
  _do_destroy "$avd"
}

cmd_list() {
  local found=0 d name
  for d in "$AVD_HOME"/*"$EPHEM_TAG"*.avd; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d" .avd)"
    found=1
    if _is_running "$name"; then printf '  %-50s [running]\n' "$name"; else printf '  %-50s [stopped]\n' "$name"; fi
  done
  [[ "$found" == "0" ]] && echo "  (no ephemeral clones)"
}

cmd_gc() {
  local d name any=0
  for d in "$AVD_HOME"/*"$EPHEM_TAG"*.avd; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d" .avd)"
    if _is_running "$name"; then
      echo ">>> [gc] skip $name (running — destroy it explicitly)"
      continue
    fi
    any=1
    rm -rf "$AVD_HOME/$name.avd" "$AVD_HOME/$name.ini" "$AVD_HOME/$name.emulog"
    echo ">>> [gc] removed $name"
  done
  [[ "$any" == "0" ]] && echo ">>> [gc] nothing to remove"
}

cmd_run() {
  local src="" console="" grpc="" headless="1" timeout="300"
  local -a cmd=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--port)  console="$2"; shift 2;;
      -g|--grpc)  grpc="$2"; shift 2;;
      --headless) headless="$2"; shift 2;;
      --timeout)  timeout="$2"; shift 2;;
      --) shift; cmd=("$@"); break;;
      *) [[ -z "$src" ]] && { src="$1"; shift; } || { echo "unexpected arg: $1 (put the command after --)" >&2; exit 1; };;
    esac
  done
  [[ -n "$src" ]] || { echo "usage: run <src> [-p PORT] [-g GRPC] -- <cmd...>" >&2; exit 1; }
  [[ ${#cmd[@]} -gt 0 ]] || { echo "ERROR: no command given after --" >&2; exit 1; }

  local clone; clone="$(_gen_clone_name "$src")"
  local EMU_PID=""
  cleanup_run() {
    local rc=$?; set +e
    [[ -n "$EMU_PID" ]] && kill -0 "$EMU_PID" 2>/dev/null && _do_destroy "$clone" || _do_destroy "$clone"
    exit $rc
  }
  trap cleanup_run EXIT INT TERM

  echo ">>> Cloning $src -> $clone" >&2
  _do_clone "$src" "$clone"
  read -r console grpc < <(_pick_ports "$console" "$grpc")
  echo ">>> Booting $clone (console $console, grpc $grpc, headless=$headless)" >&2
  local out; out=$(_do_boot "$clone" "$console" "$grpc" "$headless" 1 "$timeout")
  read -r console grpc EMU_PID <<<"$out"

  # Substitute {{CONSOLE_PORT}} / {{GRPC_PORT}} placeholders in the command's argv,
  # and also export them as env vars (for commands that read the environment instead).
  local i
  for i in "${!cmd[@]}"; do
    cmd[$i]="${cmd[$i]//\{\{CONSOLE_PORT\}\}/$console}"
    cmd[$i]="${cmd[$i]//\{\{GRPC_PORT\}\}/$grpc}"
  done
  echo ">>> Running: ${cmd[*]}" >&2
  set +e
  CONSOLE_PORT="$console" GRPC_PORT="$grpc" "${cmd[@]}"
  local rc=$?
  set -e
  echo ">>> command exited with code $rc" >&2
  exit $rc
}

# --------------------------------------------------------------------------- dispatch
main() {
  _need_tools
  local sub="${1:-}"; shift || true
  case "$sub" in
    clone)   cmd_clone   "$@";;
    boot)    cmd_boot    "$@";;
    destroy) cmd_destroy "$@";;
    list)    cmd_list    "$@";;
    gc)      cmd_gc      "$@";;
    run)     cmd_run     "$@";;
    ""|-h|--help|help)
      sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
      ;;
    *) echo "unknown subcommand: $sub (try --help)" >&2; exit 1;;
  esac
}
main "$@"
