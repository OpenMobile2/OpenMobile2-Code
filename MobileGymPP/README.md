# MobileGym++

This file is the MobileGym++ bench215 evaluation. Start the phone from the official [MobileGym-plusplus](https://github.com/OpenMobile2/MobileGym-plusplus) checkout. The same commands are in the repository [README](../README.md).

Use a different port from official MobileGym (`4172`). The Python environment is [`../AndroidWorld/environment.md`](../AndroidWorld/environment.md).

## 1. Start the frontend

`<MOBILEGYM_PP>` is that checkout. It contains `package.json` and `bench_env`.

```bash
cd <MOBILEGYM_PP>
npm install
npm run preview -- --host 127.0.0.1 --port 3000
```

Install the browser used by the eval adapter, in the `android_world` environment:

```bash
playwright install chromium
```

Leave the preview running.

## 2. Run evaluation

Pass that same directory as `-FrontendDir`. `-Port` must match the preview above.

```powershell
cd <OPENMOBILE_ROOT>\MobileGymPP
.\eval_bench215.ps1 -Runtime qwen35_thought_session -Port 3000 -Mode both `
  -FrontendDir <MOBILEGYM_PP> `
  -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName YOUR_MODEL
.\eval_bench215.ps1 -Runtime qwen3vl -Mode gui_only -Port 3000 `
  -FrontendDir <MOBILEGYM_PP> `
  -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName OpenMobile-8B
.\eval_bench215.ps1 -Runtime venus -Port 3000 `
  -FrontendDir <MOBILEGYM_PP> `
  -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName UI-Venus-1.5-8B
```

`-Runtime` is `qwen35_thought_session`, `qwen3vl`, `venus`, `gui_owl`, or `mai_ui`. `-Mode gui_only` or `-Mode hybrid` runs one side. Output is `MobileGymPP/eval_runs/<ModelName>/bench215-mock/` with `gui_only` and `hybrid` subdirectories. Pass `-OutputDir` to write somewhere else.
