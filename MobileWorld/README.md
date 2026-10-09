# MobileWorld

This file starts the backends. Evaluation commands are in the repository [README](../README.md). Start them from the official [MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) checkout.

## Windows

MobileWorld runs inside WSL.

```bash
wsl --shutdown
wsl -d ubuntu
```

## Start the backends

```bash
cd /path/to/MobileWorld
uv sync
uv run mw env run --count 2
```

`--count 2` serves `http://127.0.0.1:6800` and `http://127.0.0.1:6801`. Increase `--count` to open the next ports (`6802`, `6803`, …). Leave this process running.
