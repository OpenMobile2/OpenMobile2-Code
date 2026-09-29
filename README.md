# OpenMobile-2: Building Versatile Mobile Agents with Scalable Environments and App-Native Tools

<p align="center">
&nbsp;&nbsp;📑 <a href="<PAPER_URL>">Paper</a>&nbsp;&nbsp; | &nbsp;&nbsp;🌐 <a href="<HOMEPAGE_URL>">Homepage</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤗 <a href="https://huggingface.co/datasets/OpenMobile-2/OpenMobile-Data">Dataset</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤖 <a href="https://huggingface.co/OpenMobile-2/OpenMobile-27B">Model</a>&nbsp;&nbsp; | &nbsp;&nbsp;🤗 <a href="<DATA_VIEWER_URL>">DataViewer</a>&nbsp;&nbsp;
</p>

We introduce OpenMobile-2, a near-frontier mobile agent with fully open training environments and recipes. We make three key advances: (1) *Diverse environments with simulated commercial apps*: We build **MobileGym++**, featuring 35 realistic, functionally rich commercial-style apps with cross-app workflows, while preserving full controllability for reset and verification. Together with newly configured apps in Android emulators, this yields a diverse playground spanning over 110 apps. (2) *Open training data at scale*: Building on this foundation, we curate nearly 12K mobile interaction trajectories for supervised fine-tuning and the largest open collection of verifiable RL training data for mobile agents, comprising over 2K executable tasks with automatic rewards. (3) *Hybrid GUI and app-native tool use*: We explore an experimental mobile-use setting where apps expose selected functionalities as app-native tools alongside their GUIs, allowing agents to interleave GUI actions and tool calls within a task. We implement this setting in **MobileGym++** with over 300 carefully scoped tools across 50+ apps. Additionally, we introduce **MobileGym++ Bench** for complex, long-horizon commercial mobile scenarios, supporting both GUI-only and hybrid GUI–tool evaluation on a shared task suite.

OpenMobile-2 performs competitively across established benchmarks, including AndroidWorld (79.9) and MobileWorld (50.4), while showing promising transfer to real-device mobile use, nearly doubling SPA-Bench performance from 31.9 to 59.6.

<p align="center">
  <img src="assets/openmobile2.png" alt="OpenMobile-2" width="900">
</p>

This repository evaluates those agents. Model dialogue (prompts, history, tool-call format) lives in [`runtimes/`](runtimes/) and does not know which device it is talking to. Each benchmark only turns the parsed action into an environment step.

Set `YOUR_MODEL` and `http://<openai-compatible-host>/v1` to your OpenAI-compatible server. API keys go in the environment, not in this file.

## 📋 Contents

- [Runtimes](#runtimes)
- [AndroidWorld](#androidworld)
- [MobileGym](#mobilegym)
- [MobileWorld](#mobileworld)
- [MobileGym++ Bench](#mobilegym-bench)
- [Data synthesis](#data-synthesis)
- [Acknowledgements](#acknowledgements)
- [License](#license)

<a id="runtimes"></a>
## 🤗 Runtimes

| `--runtime` | What the model sees | Typical `last_n` |
|---|---|---|
| `qwen3vl` | One system prompt and one user prompt. Past `Action:` lines are concatenated into that user prompt, with the last N screenshots. Output is Thought, Action, and one JSON `<tool_call>`. | 1 |
| `qwen35_thought_session` | Multi-turn session used for the evaluations below. The task is sent once. Later turns are screenshots. The assistant writes `Thought:` and then one XML `<tool_call>`. Native thinking is off. | 3 |
| `venus` | Current screenshot only. Output is `<think>`, `<action>`, `<conclusion>`. | ignored |
| `gui_owl` | MobileGym / MobileGym++ official GUI-Owl adapter. The prompt stays in that benchmark. | benchmark default |

`qwen35_session` remains in code for older rollout (native thinking and API `tools=`). New evaluation commands use `qwen35_thought_session`. Format notes: [`runtimes/qwen35_thought_session.md`](runtimes/qwen35_thought_session.md).

`gui_owl` is not implemented inside `runtimes/`. Pass it as the benchmark agent name.

<a id="androidworld"></a>
## 📊 AndroidWorld

Emulator and ADB setup follow the [AndroidWorld](https://github.com/google-research/android_world) instructions. Python environment: [`AndroidWorld/environment.md`](AndroidWorld/environment.md).

```bash
cd AndroidWorld
python run.py \
  --runtime qwen35_thought_session \
  --last_n 3 \
  --console_port 5554 \
  --grpc_port 8554 \
  --perform_emulator_setup=true \
  --model_base_url http://<openai-compatible-host>/v1 \
  --model_api_key EMPTY \
  --model_name YOUR_MODEL \
  --checkpoint_dir runs/androidworld-thought \
  --task_random_seed 30
```

OpenMobile-8B uses the flat ReAct runtime:

```bash
python run.py \
  --runtime qwen3vl \
  --last_n 1 \
  --console_port 5554 \
  --grpc_port 8554 \
  --model_base_url http://<openai-compatible-host>/v1 \
  --model_api_key EMPTY \
  --model_name OpenMobile-8B \
  --checkpoint_dir runs/androidworld-qwen3vl
```

Results are written under `--checkpoint_dir`.

<a id="mobilegym"></a>
## 📊 MobileGym

Official MobileGym frontend (Playwright). Start it yourself, then point the eval at that URL.

```bash
cd "$MOBILEGYM_FRONTEND"
npm run preview -- --host 127.0.0.1 --port 4173
```

```powershell
cd <OPENMOBILE_ROOT>\MobileGym
.\run_eval_session.ps1 -Runtime qwen35_thought_session -ModelName YOUR_MODEL `
  -ModelBaseUrl http://<openai-compatible-host>/v1 -EnvUrl http://127.0.0.1:4173 -Port 4173
.\run_eval_session.ps1 -Runtime qwen3vl -ModelName OpenMobile-8B `
  -ModelBaseUrl http://<openai-compatible-host>/v1
.\run_eval_session.ps1 -Runtime venus -ModelName UI-Venus-1.5-8B `
  -ModelBaseUrl http://<openai-compatible-host>/v1
.\run_eval_session.ps1 -Runtime gui_owl -ModelName GUI-Owl-1.5-8B `
  -ModelBaseUrl http://<openai-compatible-host>/v1
```

`-Runtime` is `qwen35_thought_session` (default, last_n 3), `qwen3vl` (last_n 1), `venus` (last_n 1), or `gui_owl`. `qwen35_session` is still accepted. Output is `MobileGym/runs_main/<ModelName>` with dots removed. `qwen3vl` writes `runs_main/qwen3vl/<ModelName>` so it does not resume an older folder of the same model name. An existing `meta.json` resumes unfinished tasks.

<a id="mobileworld"></a>
## 📊 MobileWorld

HTTP backends, one host per worker. GUI tasks and MCP tasks are separate runs. Do not put both on the same hosts at the same time.

```bash
cd /path/to/MobileWorld
uv run mw env run --count 2
```

```powershell
cd <OPENMOBILE_ROOT>\MobileWorld
python parallel_eval_mw.py `
  --hosts http://127.0.0.1:6800,http://127.0.0.1:6801 `
  --runtime qwen35_thought_session `
  --last_n 3 `
  --tasks ALL `
  --output_dir eval-gui/YOUR_MODEL/gui-only `
  --qwen3vl_model_base_url http://<openai-compatible-host>/v1 `
  --qwen3vl_model_name YOUR_MODEL `
  --qwen3vl_model_api_key EMPTY
```

MCP tasks use the same runtime and add `--mcp_only`:

```powershell
python parallel_eval_mw.py `
  --hosts http://127.0.0.1:6800,http://127.0.0.1:6801 `
  --runtime qwen35_thought_session `
  --last_n 3 `
  --mcp_only `
  --enable_user_interaction `
  --output_dir eval-mcp/YOUR_MODEL `
  --qwen3vl_model_base_url http://<openai-compatible-host>/v1 `
  --qwen3vl_model_name YOUR_MODEL `
  --qwen3vl_model_api_key EMPTY
```

Summaries are `eval_summary.json` inside each output directory. Flat OpenMobile-8B is `--runtime qwen3vl --last_n 1` on the GUI run (MCP expects a session runtime).

<a id="mobilegym-bench"></a>
## 📊 MobileGym++ Bench

bench215 on the mock frontend: `gui_only` and `hybrid`. Start the mock on a different port from official MobileGym.

```powershell
cd <MOBILEGYM_MOCK_ROOT>\trial_apps\mobilegym
npm run preview -- --host 127.0.0.1 --port 4173
```

```powershell
cd <OPENMOBILE_ROOT>\MobileGymPP
.\eval_bench215.ps1 -Runtime qwen35_thought_session -Port 4173 -Mode both `
  -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName YOUR_MODEL
.\eval_bench215.ps1 -Runtime qwen3vl -Mode gui_only -Port 4173 `
  -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName OpenMobile-8B
.\eval_bench215.ps1 -Runtime venus -Port 4173 `
  -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName UI-Venus-1.5-8B
```

`-Runtime` is `qwen35_thought_session`, `qwen3vl`, `venus`, `gui_owl`, or `mai_ui`. `-Mode gui_only` or `-Mode hybrid` runs one side. Output is `MobileGymPP/eval_runs/<ModelName>/bench215-mock/` with `gui_only` and `hybrid` subdirectories. Pass `-OutputDir` to write somewhere else.

<a id="data-synthesis"></a>
## 🎮 Data synthesis

Task synthesis is [`task_synthesis/`](task_synthesis/). AndroidWorld, MobileWorld, and MobileGym each have an explore / rollout pipeline (`run_diy.py`, `run_diy_mw.py`, `run_diy_mg.py`) that calls the same `runtimes/` presets. Those paths are not required to run the four evaluations.

Local eval dumps (`runs_main/`, `eval-gui/`, `eval-mcp/`, `eval_runs/`, and the other directories in [`.gitignore`](.gitignore)) stay on disk and are not part of the release.

<a id="acknowledgements"></a>
## 💐 Acknowledgements

[AndroidWorld](https://github.com/google-research/android_world)&#8194;
[MobileWorld](https://github.com/Tongyi-MAI/MobileWorld)&#8194;
[MobileGym](https://github.com/Purewhiter/mobilegym)&#8194;
[Qwen-VL](https://github.com/QwenLM/Qwen3-VL)&#8194;
[LlamaFactory](https://github.com/hiyouga/LlamaFactory)

<a id="license"></a>
## ⚖️ License

Apache 2.0. See [LICENSE](LICENSE). Third-party models, datasets, and benchmark code keep their own licenses.
