#!/usr/bin/env python3
"""ephemeral_avd.py — standalone throw-away AVD clones for any Python caller.

独立、可复用的一次性 AVD 克隆组件。到处都要用 copy 出来的 AVD 时，import 这个模块即可；
底层复用同目录下的 avd_ephemeral.sh（单一真源：APFS 写时复制克隆 + 改名 + 选端口 + 销毁），
不重复实现那套逻辑。源 AVD 全程不动。

核心：一个上下文管理器 `ephemeral_avd()` —— 克隆 -> 启动克隆 -> 交出端口 -> **退出时必删**
（正常返回 / 抛异常 / Ctrl-C / SIGTERM 都会清理）。

Python 用法（推荐）——包住你现有的 env 初始化即可：

    from ephemeral_avd import ephemeral_avd
    from android_world.env import env_launcher

    with ephemeral_avd("MineAndroid-general-user") as avd:
        env = env_launcher.load_and_setup_env(
            console_port=avd.console_port, grpc_port=avd.grpc_port)
        ...            # 跑 eval / walk / 手动实验，随便造
    # 到这里克隆已被销毁，源 AVD 未受影响

也可分步用：

    from ephemeral_avd import clone, boot, destroy
    name = clone("MineAndroid-general-user")          # 秒级，什么都没启动
    console_port, grpc_port = boot(name)              # 启动并等开机完成
    try:
        ...
    finally:
        destroy(name)

命令行——与 avd_ephemeral.sh 完全一致的子命令（透传）：

    python ephemeral_avd.py clone   <src>
    python ephemeral_avd.py boot    <name> --wait
    python ephemeral_avd.py destroy <name>
    python ephemeral_avd.py list
    python ephemeral_avd.py gc                         # 清理残留（SIGKILL 后的兜底）
    python ephemeral_avd.py run     <src> -- <cmd...>

残留兜底：若进程被 `kill -9`（无法捕获）导致克隆没删掉，跑一次 `gc()` /
`python ephemeral_avd.py gc` 即可清掉所有未运行的 *-ephem-* 克隆。
"""

from __future__ import annotations

import atexit
import contextlib
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

# The shell script is the single source of truth for the clone/boot/destroy logic.
_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avd_ephemeral.sh")

# Clones this process created and still owns — used by the atexit safety net.
_LIVE_CLONES: "set[str]" = set()


class EphemeralAvdError(RuntimeError):
    """avd_ephemeral.sh failed or returned something unparseable."""


def _run(args: List[str], capture: bool = True) -> str:
    """Invoke `avd_ephemeral.sh <args>`. stderr streams live (its `>>> ...` progress);
    stdout is captured when `capture` so callers can parse names/ports from it."""
    if not os.path.exists(_SCRIPT):
        raise EphemeralAvdError(
            f"avd_ephemeral.sh not found next to ephemeral_avd.py: {_SCRIPT}"
        )
    proc = subprocess.run(
        [_SCRIPT, *args],
        stdout=subprocess.PIPE if capture else None,
        stderr=None,  # inherit — let the shell's progress reach the terminal
        text=True,
    )
    if proc.returncode != 0:
        raise EphemeralAvdError(
            f"`avd_ephemeral.sh {' '.join(args)}` failed (exit {proc.returncode})"
        )
    return proc.stdout or ""


def clone(src: str, name: Optional[str] = None) -> str:
    """Clone source AVD `src` (copy-on-write). Returns the clone's AVD name.

    Nothing is booted. The returned name is tracked so it is cleaned up on
    interpreter exit even if you forget to call `destroy`.
    """
    out = _run(["clone", src, *([name] if name else [])])
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise EphemeralAvdError("clone did not print a clone name on stdout")
    clone_name = lines[-1]
    _LIVE_CLONES.add(clone_name)
    return clone_name


def boot(
    name: str,
    console_port: Optional[int] = None,
    grpc_port: Optional[int] = None,
    headless: bool = True,
    timeout: int = 300,
) -> Tuple[int, int]:
    """Boot AVD `name` (typically a clone) and block until boot_completed.

    Ports are auto-picked (free-port scan) when omitted, so many clones can run
    in parallel without colliding. Returns (console_port, grpc_port).
    """
    args = ["boot", name, "--headless", "1" if headless else "0",
            "--timeout", str(timeout), "--wait"]
    if console_port:
        args += ["-p", str(console_port)]
    if grpc_port:
        args += ["-g", str(grpc_port)]
    out = _run(args)
    m = re.search(r"CONSOLE_PORT=(\d+)\s+GRPC_PORT=(\d+)", out)
    if not m:
        raise EphemeralAvdError(f"could not parse ports from boot output: {out!r}")
    return int(m.group(1)), int(m.group(2))


def destroy(name: str) -> None:
    """Kill the AVD if running, then delete the clone dir + its .ini."""
    try:
        _run(["destroy", name], capture=False)
    finally:
        _LIVE_CLONES.discard(name)


def gc() -> None:
    """Destroy ALL not-running '*-ephem-*' clones — the backstop for leaked clones."""
    _run(["gc"], capture=False)


@dataclass
class Clone:
    """A booted, ready-to-use ephemeral clone."""
    name: str
    console_port: int
    grpc_port: int


@contextlib.contextmanager
def ephemeral_avd(
    src: str,
    console_port: Optional[int] = None,
    grpc_port: Optional[int] = None,
    headless: bool = True,
    timeout: int = 300,
    keep: bool = False,
) -> Iterator[Clone]:
    """Clone `src`, boot the clone, yield a `Clone`, and ALWAYS destroy it on exit.

    Destroy runs on normal return, on exception, on Ctrl-C, and on SIGTERM (a
    scoped handler makes SIGTERM raise so `finally` runs). SIGKILL cannot be
    caught — use `gc()` to sweep leftovers.

    Args:
      src: name of the golden/source AVD to clone (never mutated).
      console_port / grpc_port: pin ports, or leave None to auto-pick free ones.
      headless: run the emulator with -no-window.
      timeout: seconds to wait for boot_completed.
      keep: if True, do NOT destroy on exit (for debugging a failed run).
    """
    name = clone(src)

    # Scope a SIGTERM handler so an external `kill` still triggers cleanup.
    prev_handler = None
    installed = False
    try:
        def _on_sigterm(signum, frame):  # noqa: ANN001
            raise KeyboardInterrupt()  # unwinds to the finally below
        prev_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _on_sigterm)
        installed = True
    except (ValueError, OSError):
        # signal.signal only works in the main thread; skip otherwise.
        pass

    try:
        cport, gport = boot(name, console_port, grpc_port, headless, timeout)
        yield Clone(name=name, console_port=cport, grpc_port=gport)
    finally:
        if installed:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGTERM, prev_handler)
        if keep:
            print(f">>> [ephemeral_avd] keeping clone {name} (keep=True); "
                  f"remove later with: python ephemeral_avd.py destroy {name}",
                  file=sys.stderr)
            _LIVE_CLONES.discard(name)  # caller owns it now
        else:
            destroy(name)


@atexit.register
def _cleanup_leftovers() -> None:
    """Last-resort sweep of clones this process created but never destroyed."""
    for name in list(_LIVE_CLONES):
        with contextlib.suppress(Exception):
            destroy(name)


def _main(argv: List[str]) -> int:
    """CLI parity: forward every arg straight to avd_ephemeral.sh."""
    if not os.path.exists(_SCRIPT):
        print(f"avd_ephemeral.sh not found: {_SCRIPT}", file=sys.stderr)
        return 1
    return subprocess.call([_SCRIPT, *argv[1:]])


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
