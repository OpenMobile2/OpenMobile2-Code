# MobileGym

This file is the MobileGym evaluation. Start the frontend from the official [mobilegym](https://github.com/Purewhiter/mobilegym) checkout. The same commands are in the repository [README](../README.md).

The Python environment is [`../AndroidWorld/environment.md`](../AndroidWorld/environment.md).

## 1. Start the frontend

`<MOBILEGYM_FRONTEND>` is the checkout that contains `package.json` and `bench_env`. In the official repo that is the inner `mobilegym/` directory.

```bash
cd <MOBILEGYM_FRONTEND>
npm install
npm run preview -- --host 127.0.0.1 --port 4172
```

Install the browser used by the eval adapter, in the `android_world` environment:

```bash
playwright install chromium
```

Leave the preview running.

## 2. Run evaluation

Pass that same directory as `-FrontendDir`. `-EnvUrl` and `-Port` must match the preview above.

```powershell
cd <OPENMOBILE_ROOT>\MobileGym
.\run_eval_session.ps1 -Runtime qwen35_thought_session -ModelName YOUR_MODEL `
  -FrontendDir <MOBILEGYM_FRONTEND> `
  -ModelBaseUrl http://<openai-compatible-host>/v1 -EnvUrl http://127.0.0.1:4172 -Port 4172
.\run_eval_session.ps1 -Runtime qwen3vl -ModelName OpenMobile-8B `
  -FrontendDir <MOBILEGYM_FRONTEND> `
  -ModelBaseUrl http://<openai-compatible-host>/v1
.\run_eval_session.ps1 -Runtime venus -ModelName UI-Venus-1.5-8B `
  -FrontendDir <MOBILEGYM_FRONTEND> `
  -ModelBaseUrl http://<openai-compatible-host>/v1
.\run_eval_session.ps1 -Runtime gui_owl -ModelName GUI-Owl-1.5-8B `
  -FrontendDir <MOBILEGYM_FRONTEND> `
  -ModelBaseUrl http://<openai-compatible-host>/v1
```

`-Runtime` is `qwen35_thought_session` (default, last_n 3), `qwen3vl` (last_n 1), `venus` (last_n 1), or `gui_owl`. Output is `MobileGym/runs_main/<ModelName>` with dots removed.
