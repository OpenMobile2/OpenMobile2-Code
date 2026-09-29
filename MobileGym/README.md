# MobileGym backend for OpenMobile data synthesis

Evaluation commands are in the repository [README](../README.md). This file is the data-synthesis pipeline.

GUI-only official eval and freeform DIY live here. To roll **hybrid GUI + App
Tools** on `hybrid_tools_custom_v4` / `v5`, use the sibling
[`../MobileGymPP/`](../MobileGymPP/).

This folder adapts **MobileGym** (Playwright SPA simulator) to the OpenMobile
pipeline that was originally built on AndroidWorld / MobileWorld.

**Agents:**
| Path | How | Example |
|------|-----|---------|
| Official SR eval | `bench_env --agent` | `run_eval_session.ps1 -Runtime qwen35_thought_session` / `qwen3vl` / `venus` / `gui_owl` |
| DIY / pipeline (ReAct stack) | `agent_factory --runtime` + `mg_env` | baseline `qwen3vl`; Venus: `--runtime venus` / `run_diy_venus.ps1` |

Venus uses official `<think>/<action>/<conclusion>` (current image only, `last_n=1`), not Qwen JSON tool_call.

```text
MobileGym frontend (npm run dev / preview / nginx)
        │  Playwright (screenshot + __SIM_INPUT__ / __OS__)
        ▼
   mg_env.MobileGymEnv   ← AsyncEnv-compatible adapter
        │
        ├── prepare_tasks.py     → prepared_tasks.json   [from tasks/]
        ├── run_diy_mg.py        → runs/.../             [same layout as AW run_diy]
        ├── parallel_rollout_mg.py
        ├── process_trajs / process_refine               [reuse MW/AW scripts]
        └── run_full_pipeline_mg.py                      [one-click orchestrator]
```

## Key difference vs MobileWorld

| | MobileWorld | MobileGym |
|---|---|---|
| Explore / synthesize | `random_walk` → task_synthesis | **Skipped** — use prebuilt `tasks/` |
| Env transport | HTTP `/screenshot` `/step` | Playwright + `__SIM__` / `__OS__` |
| Task type | Registered MW tasks + params | Freeform `MobileGymFreeform` |

Prebuilt grounded tasks live in [`tasks/`](tasks/) (`tasks_5000.jsonl`,
`tasks_easy/medium/hard.jsonl`). Instructions are English; entity names match
real MobileGym seed data (often Chinese UI strings).

**Mock-safe generation:** do **not** treat entity lists in `defaults.json` as
proof a goal is completable. See
[`tasks/TASK_GENERATION_GUIDE.md`](tasks/TASK_GENERATION_GUIDE.md) for how to
build tasks from `navigation.declaration.ts` + bindTap + reachable nav graphs
(+ optional `bench_env`). The current `generate_tasks.py` is still largely
entity-template based; regenerate only after following that guide.

## Prerequisites

1. MobileGym frontend running, e.g.:

```bash
cd /path/to/mobilegym-mock/trial_apps/mobilegym
npm install
# optional: extract mobilegym-data for media/themes
npm run dev   # http://localhost:3000
```

2. OpenMobile AndroidWorld Python env (for `android_world` + agents), see
   [`../AndroidWorld/environment.md`](../AndroidWorld/environment.md).

3. Playwright Chromium for the adapter:

```bash
playwright install chromium
```

4. Model API for rollout / refine:

```bash
export OPENAI_BASE_URL=...
export OPENAI_API_KEY=...
```

## One-click pipeline

Frontend first (in `mobilegym-mock/trial_apps/mobilegym`):

```bash
# ≤8 browsers: vite preview is fine
npm run build && npm run preview -- --port 4172

# ≥8 browsers (optional, more stable static serving): nginx gateway
./scripts/server/start_nginx_gateway.sh   # → https://localhost:4180
```

Then run the pipeline. **Parallelism is Playwright workers, not frontend URL count.**
One URL is enough: each worker keeps its own SPA state in its own browser.

```bash
cd <OPENMOBILE_ROOT>/MobileGym

# 8-way parallel against a single preview URL
python run_full_pipeline_mg.py \
  --env_url http://127.0.0.1:4173 \
  --parallel 8 \
  --tasks_input tasks/tasks_5000.jsonl \
  --use_memgui_prompt \
  --qwen3vl_model_base_url "https://<openai-compatible-host>/v1" \
  --qwen3vl_model_name gemini-3.1-pro-preview \
  --qwen3vl_model_api_key "YOUR_API_KEY"
```

Or call the rollout launcher directly:

```bash
python parallel_rollout_mg.py \
  --env_url http://127.0.0.1:4173 \
  --parallel 8 \
  --input_json prepared_tasks.json \
  --output_dir runs/mobilegym_rollout \
  --agent_name qwen3vl \
  --use_memgui_prompt \
  --qwen3vl_model_base_url ... --qwen3vl_model_name ... --qwen3vl_model_api_key ...
```

Stages:

```text
prepare_tasks → rollout → merge → refine_conclusion → refine_thinking
```

Useful flags:

```bash
# Smoke: 20 easy tasks, stop after prepare
python run_full_pipeline_mg.py \
  --env_url http://127.0.0.1:4173 \
  --difficulty easy --limit 20 \
  --stop_after prepare_tasks

# Resume an existing run from rollout
python run_full_pipeline_mg.py \
  --run_id 20260803_150000 \
  --start_from rollout \
  --env_url http://127.0.0.1:4173 \
  --parallel 8 \
  --use_memgui_prompt \
  --qwen3vl_model_base_url ... --qwen3vl_model_name ...
```

Outputs:

```text
MobileGym/pipeline_runs/<run_id>/
  synthesis/prepared_tasks.json
  rollout/
    <sample_id>_MobileGymFreeform/
    data_merge_success.json
    data_merge_success_conclusion.json
    data_merge_success_conclusion_thinking.json
  run_config.json
```

After `refine_thinking`, success data is uploaded automatically to
`/mnt/afs/<user>/data_gym/<NNNN>_<run_id>/`
(thinking json + success trajectory folders). Use `--no_upload` to skip, or re-upload a finished run:

```bash
python upload_success_to_remote.py --run_id 20260803_194544
python upload_success_to_remote.py --run_id 20260803_194544 --dry_run
```

## Manual stage-by-stage

### 1. Prepare tasks

```bash
python prepare_tasks.py \
  --input tasks/tasks_hard.jsonl \
  --output /tmp/prepared_tasks.json \
  --limit 100
```

### 2. Rollout

```bash
# Baseline: qwen3vl ReAct (Thought/Action + JSON tool_call)
python run_diy_mg.py \
  --input_json /tmp/prepared_tasks.json \
  --output_dir runs/mobilegym_rollout \
  --env_url http://127.0.0.1:3000 \
  --agent_name qwen3vl \
  --use_memgui_prompt=true \
  --qwen3vl_model_base_url http://YOUR_VLLM/v1 \
  --qwen3vl_model_name YOUR_MODEL

# UI-Venus-1.5 (same mg_env + agent_factory path; native Venus prompt)
python run_diy_mg.py \
  --input_json /tmp/prepared_tasks.json \
  --output_dir runs/UI-Venus-1.5-8B/diy \
  --env_url http://127.0.0.1:4172 \
  --runtime venus \
  --last_n 1 \
  --qwen3vl_model_base_url http://YOUR_VLLM/v1 \
  --qwen3vl_model_name UI-Venus-1.5-8B

# Or: .\run_diy_venus.ps1 -InputJson ... -EnvUrl http://127.0.0.1:4172
```

### Official scoring eval (bench_env, not DIY)

```powershell
.\run_eval_session.ps1 -Runtime venus `
  -ModelBaseUrl http://YOUR_VLLM/v1 `
  -ModelName UI-Venus-1.5-8B `
  -Port 4172
```

Parallel (multiple frontend URLs / shards):

```bash
python parallel_rollout_mg.py \
  --env_urls http://127.0.0.1:3000,http://127.0.0.1:3001 \
  --input_json /tmp/prepared_tasks.json \
  --output_dir runs/mobilegym_rollout \
  --agent_name qwen3vl \
  --use_memgui_prompt \
  --qwen3vl_model_base_url http://YOUR_VLLM/v1 \
  --qwen3vl_model_name YOUR_MODEL
```

### 3. Post-process → SFT

```bash
python process_trajs.py \
  --runs_dir runs/mobilegym_rollout \
  --output-name data_merge_success.json

python process_refine.py \
  --input runs/mobilegym_rollout/data_merge_success.json \
  --output runs/mobilegym_rollout/data_merge_success_conclusion.json \
  --mode conclusion

python process_refine.py \
  --input runs/mobilegym_rollout/data_merge_success_conclusion.json \
  --output runs/mobilegym_rollout/data_merge_success_conclusion_thinking.json \
  --mode thinking
```

Optional ShareGPT conversion (reuse AW script):

```bash
cd ../AndroidWorld
python convert_traj.py \
  --input ../MobileGym/runs/mobilegym_rollout/data_merge_success_conclusion_thinking.json \
  --output ../MobileGym/runs/mobilegym_rollout/sft.json
```

## Prompt modes

| Mode | Flags |
|------|--------|
| Default Qwen3VL | `--agent_name qwen3vl` |
| MemGUI | `--agent_name qwen3vl --use_memgui_prompt` |
| MemGUI + Qwen3.5 XML | `--agent_name qwen35vl --use_memgui_prompt --memgui_prompt_format qwen35` |

## Notes

- Freeform tasks have **no state judge**. Success is whatever the agent declares via `terminate` / `status` (same as OpenMobile DIY freeform semantics after merge filters).
- Entity names inside instructions are grounded in MobileGym seed data; do not invent contacts/products.
- Ensure `mobilegym-data/` is present if tasks rely on CDN images/themes.
