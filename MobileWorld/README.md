# MobileWorld backend for OpenMobile data synthesis

Evaluation commands are in the repository [README](../README.md). This file is the data-synthesis pipeline.

This folder adapts **MobileWorld** (screenshot / XML / step over HTTP) to the
OpenMobile pipeline that was originally built on AndroidWorld.

```text
MobileWorld Docker (mw env run)
        │  HTTP /screenshot /xml /step /task/*
        ▼
   mw_env.MobileWorldEnv   ← AsyncEnv-compatible adapter
        │
        ├── random_walk_mw.py   → explore_results/  (same layout as AW)
        ├── process_explore.py  → state_transfer_explore.json   [reuse AW script]
        ├── task_synthesis/     → synthesized_tasks_*_final.json [reuse]
        ├── run_diy_mw.py       → runs/.../  (same layout as AW run_diy)
        └── process_trajs / convert_traj     [reuse AW scripts]
```

The only intentional difference vs AndroidWorld is **how** screenshots / UI
trees / actions are obtained. Task synthesis and SFT conversion stay unchanged.

## Windows

1.安装wsl
2. 
```bash
wsl --shutdown
wsl -d ubuntu

```

## Prerequisites

1. MobileWorld image + backends:

```bash
cd /path/to/MobileWorld
uv sync
uv run mw env run --count 1   # backend http://127.0.0.1:6800
```

2. OpenMobile AndroidWorld Python env (for `android_world` + agents), see
   [`../AndroidWorld/environment.md`](../AndroidWorld/environment.md).

3. Strong-model API for text synthesis / optional editable-field typing:

```bash
export OPENAI_BASE_URL=...
export OPENAI_API_KEY=...
```

## 1. Explore

```bash
cd /path/to/OpenMobile-Code/AndroidWorld

python ../MobileWorld/random_walk_mw.py \
  --aw_host http://127.0.0.1:6800 \
  --output_dir ../MobileWorld/explore_results \
  --num_step 10 \
  --max_tasks 20
```

Outputs:

```text
MobileWorld/explore_results/
  screenshots/
  trajectories/
  params/
  unclickable_elem_pool/
```

Convert to state-transfer format (reuse AW script):

```bash
python process_explore.py \
  --traj_dir ../MobileWorld/explore_results/trajectories \
  --out ../MobileWorld/explore_results/state_transfer_explore.json
```

## 2. Synthesize tasks

```bash
cd /path/to/OpenMobile-Code/task_synthesis

python pipeline.py \
  --dataset_id mobileworld_explore \
  --state_transfer ../MobileWorld/explore_results/state_transfer_explore.json \
  --screenshots_dir ../MobileWorld/explore_results/screenshots \
  --prompt_suite mobileworld \
  --max_num_syn_screen 1000 \
  --max_workers 64
```

Use `--prompt_suite mobileworld` so synthesis/judge prompts match MW apps
(Mail / Messages / Mastodon / Mattermost / Taodian / Calendar, …) instead of
AndroidWorld Broccoli/Markor examples. Prompts live in
[`../task_synthesis/prompts_mobileworld.py`](../task_synthesis/prompts_mobileworld.py).

Final file:

```text
task_synthesis/output/mobileworld_explore/synthesized_tasks_mobileworld_explore_final.json
```

## 3. Rollout

Prompt / runtime is selected by `--runtime` (preferred) or legacy flags:

| Mode | Flags |
|------|--------|
| **Session (recommended)** | `--runtime qwen35_thought_session --last_n 3` (XML tool_call) |
| Session + Qwen3-VL JSON | `--runtime qwen3vl_session --last_n 3` |
| UI-Venus-1.5 | `--runtime venus` (current image only; ignores `--last_n`) |
| Default flat Qwen3VL | `--agent_name qwen3vl` |
| MemGUI (legacy flat) | `--agent_name qwen3vl --use_memgui_prompt=true` |
| MemGUI + Qwen3.5 XML | `--agent_name qwen35vl --use_memgui_prompt=true --memgui_prompt_format=qwen35` |
| Memory baseline (legacy) | `--agent_name qwen3vl --use_memory_prompt=true` |

Session runtimes write `result.json` as `{id, runtime, goal, messages, images, done}` (see [`../runtimes/qwen35_thought_session.md`](../runtimes/qwen35_thought_session.md)).  
`--last_n`: keep screenshots only in the last N image-bearing turns (default 1).  
`--enable_thinking`: session only (Qwen/vLLM).  
`--reasoning_effort`: Kimi K3 only (`low` / `high` / `max`).

**Gemini** (`gemini-*`): session uses text
`<thinking>` + JSON `<tool_call>` (no `<conclusion>`), with screenshots as follow-up
**user** turns. No native `thinking_config` / `reasoning_content`.

**Kimi K3**: native `tool_calls` + `reasoning_content` (Preserved Thinking).

Legacy flat modes: `memgui_prompt_format` = `auto` / `qwen3vl` / `qwen35`;  
`use_memgui_prompt` and `use_memory_prompt` are mutually exclusive.

```bash
cd /path/to/OpenMobile-Code/AndroidWorld

python ../MobileWorld/run_diy_mw.py \
  --input_json ../task_synthesis/output/mobileworld_explore/synthesized_tasks_mobileworld_explore_final.json \
  --output_dir ../MobileWorld/runs/mobileworld_explore_rollout \
  --params_dir ../MobileWorld/explore_results/params \
  --aw_host http://127.0.0.1:6800 \
  --runtime qwen35_thought_session \
  --last_n 3 \
  --enable_thinking true \
  --qwen3vl_model_base_url https://<openai-compatible-host>/v1 \
  --qwen3vl_model_name YOUR_API_KEY \
  --qwen3vl_model_api_key EMPTY
```

Policy-switching (same flags as AW `run_diy.py`):

```bash
python ../MobileWorld/run_diy_mw.py \
  ... \
  --agent_name qwen3vl_switching \
  --qwen3vl_switching_weak_model_base_url http://WEAK/v1 \
  --qwen3vl_switching_weak_model_name WEAK_MODEL
```

## 4. Post-process → SFT

**Session runtime** (`qwen35_thought_session` / `qwen3vl_session`): each trajectory folder already contains
session-format `result.json` (`{id, messages, images, done}`). Collect those files directly for
training; skip merge/refine/convert below.

**Legacy flat runtime** (memgui / qwen3vl without session):

```bash
cd /path/to/OpenMobile-Code/AndroidWorld

python process_trajs.py \
  --runs_dir ../MobileWorld/runs/mobileworld_explore_rollout \
  --output-name data_merge_success.json

python process_refine.py \
  --input ../MobileWorld/runs/mobileworld_explore_rollout/data_merge_success.json \
  --output ../MobileWorld/runs/mobileworld_explore_rollout/data_merge_success_conclusion_thinking.json \
  --mode thinking

python convert_traj.py \
  --input ../MobileWorld/runs/mobileworld_explore_rollout/data_merge_success_conclusion_thinking.json \
  --output ../MobileWorld/runs/mobileworld_explore_rollout/openmobile_train_mw.json \
  --refine \
  --model-format qwen3vl
```

## Key files

| File | Role |
|------|------|
| `mw_env.py` | HTTP adapter: screenshot/XML/step ↔ AsyncEnv |
| `random_walk_mw.py` | Exploration |
| `run_diy_mw.py` | Instruction rollout with OpenMobile agents |
| `parallel_explore_mw.py` | Multi-backend explore launcher |
| `parallel_rollout_mw.py` | Multi-backend rollout launcher |
| `run_full_pipeline_mw.py` | Full pipeline: explore → synth → rollout → refine |

## Parallel rollout

One MobileWorld container ≈ one rollout worker. Share the same `output_dir`;
samples are sharded by `sample_id` hash so workers never collide. Existing
trajectory dirs are skipped (safe to restart).

```bash
# In MobileWorld repo / WSL
uv run mw env run --count 4
# backends: 6800, 6801, 6802, 6803
```

```powershell
conda activate android_world
cd <OPENMOBILE_ROOT>\MobileWorld

python parallel_rollout_mw.py `
  --hosts http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802,http://127.0.0.1:6803 `
  --input_json ../task_synthesis/output/mobileworld_explore/synthesized_tasks_mobileworld_explore_final.json `
  --output_dir runs/mobileworld_explore_rollout `
  --params_dir explore_results/params `
  --runtime qwen35_thought_session `
  --last_n 3 `
  --qwen3vl_model_base_url http://YOUR_VLLM/v1 `
  --qwen3vl_model_name YOUR_MODEL
```

Or manually:

```powershell
python run_diy_mw.py --aw_host http://127.0.0.1:6800 --shard_index 0 --num_shards 4 ...
python run_diy_mw.py --aw_host http://127.0.0.1:6801 --shard_index 1 --num_shards 4 ...
```

Note: each Android emulator is heavy. On ~32GB RAM / WSL 16–20GB, start with `--count 2`.
Also ensure the VLM endpoint can handle concurrent requests from all workers.


## One-shot full pipeline (recommended)

Parallel explore uses the same sharding as `parallel_explore_mw.py`
(`--shard_index` / `--num_shards` per host). Everything from explore through
`data_merge_success_conclusion_thinking.json` is wrapped in one script, with
**isolated directories per run**:

```text
MobileWorld/pipeline_runs/<run_id>/
  explore/                  # random_walk outputs + state_transfer
  synthesis/                # task_synthesis outputs (*_final.json)
  rollout/                  # DIY trajectories + merged/refined JSON
  run_config.json
```

```powershell
conda activate android_World
cd <OPENMOBILE_ROOT>\MobileWorld

# WSL first: uv run mw env run --count 3

python run_full_pipeline_mw.py `
  --hosts http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802 `
  --num_step 10 `
  --runtime qwen35_thought_session `
  --last_n 3 `
  --stop_after rollout `
  --qwen3vl_model_base_url https://<openai-compatible-host>/v1 `
  --qwen3vl_model_name gemini-3.1-pro-preview `
  --qwen3vl_model_api_key YOUR_KEY `
  --openai_base_url https://<openai-compatible-host>/v1 `
  --openai_api_key YOUR_KEY `
  --openai_model gemini-3.1-pro-preview
```

Session rollout writes `rollout/<sample_id>_<task>/result.json` directly. Use `--stop_after rollout`
to skip legacy merge/refine stages. For flat memgui trajectories, omit `--stop_after rollout` and
run through `refine_thinking` as before.

Omit `--run_id` to auto-name with timestamp (`20260803_143000`). Re-run with the
same `--run_id` and `--start_from` to resume:

```powershell
python run_full_pipeline_mw.py --run_id 20260803_143000 --start_from rollout --hosts ...
python run_full_pipeline_mw.py --run_id 20260803_143000 --start_from merge --stop_after refine_thinking
```

Stages: `explore → process_explore → synthesize → rollout` (session)  
or `explore → … → rollout → merge → refine_conclusion → refine_thinking` (legacy flat).

After the pipeline finishes, if `data_merge_success_conclusion_thinking.json` exists it is
uploaded automatically (with all success trajectory folders) to
`<user>@<gateway-host>:/mnt/afs/<user>/data_mobile/<NNNN>_<run_id>` (SSH port `<ssh-port>`).
Use `--no_upload` to skip, or re-upload a finished run manually:

```powershell
python upload_success_to_remote.py --run_id 20260803_171524
python upload_success_to_remote.py --run_id 20260803_171524 --dry_run
```


## Parallel explore (manual)

One MobileWorld container ≈ one explorer. Launch several backends, then shard tasks:

```bash
# In MobileWorld repo / WSL
uv run mw env run --count 4
# backends: 6800, 6801, 6802, 6803
```

```powershell
conda activate android_world
cd <OPENMOBILE_ROOT>\MobileWorld
python parallel_explore_mw.py `
  --hosts http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802,http://127.0.0.1:6803 `
  --num_step 10
```

Or manually:

```powershell
python random_walk_mw.py --aw_host http://127.0.0.1:6800 --shard_index 0 --num_shards 4
python random_walk_mw.py --aw_host http://127.0.0.1:6801 --shard_index 1 --num_shards 4
# ...
```

Note: each Android emulator is heavy. On ~32GB RAM / WSL 16–20GB, start with `--count 2`.


- UI tree comes from MobileWorld `/xml` (UIAutomator), converted with AW
  `xml_dump_to_ui_elements`. Quality may differ slightly from AW a11y forest.
- MobileWorld task init does not replay AW-style `generate_random_params`;
  `params/*.pkl` mainly store seed + task metadata for bookkeeping. Init still
  goes through MW `/task/init` (snapshot/state inside the container).
- Prefer GUI-only tasks first (`--enable_mcp=false`). MCP / ask_user tasks need
  extra keys and agent support.
- Keep WSL/Docker memory headroom; one MobileWorld container is usually enough
  for exploration/rollout debugging.



# 一键式总流程：

1. 启动wsl并在里面启动ubuntu
```bash
wsl --shutdown
wsl -d ubuntu
cd MobileWorld
uv run mw env run --count 1 --launch-interval 20  #实测开3个docker并行是比较稳定的
```
2. 接着另起一个终端
```bash
conda activate android_world
cd <OPENMOBILE_ROOT>\MobileWorld

python run_full_pipeline_mw.py `
  --hosts http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802 `
  --runtime qwen35_thought_session `
  --last_n 3 `
  --qwen3vl_model_base_url https://<openai-compatible-host>/v1 `
  --qwen3vl_model_name gemini-3.1-pro-preview `
  --qwen3vl_model_api_key YOUR_API_KEY `
  --openai_base_url https://<openai-compatible-host>/v1 `
  --openai_api_key YOUR_API_KEY `
  --openai_model gemini-3.1-pro-preview
```
断点续跑
```bash
python run_full_pipeline_mw.py --run_id 20260803_150713 --start_from rollout `
  --hosts http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802 `
  --runtime qwen35_thought_session `
  --last_n 3 `
  --qwen3vl_model_base_url https://<openai-compatible-host>/v1 `
  --qwen3vl_model_name gemini-3.1-pro-preview `
  --qwen3vl_model_api_key YOUR_API_KEY `
  --openai_base_url https://<openai-compatible-host>/v1 `
  --openai_api_key YOUR_API_KEY `
  --openai_model gemini-3.1-pro-preview `
  --no_upload
```
流水线跑完后，若存在 `data_merge_success_conclusion_thinking.json`，会自动把该 JSON
和所有成功轨迹文件夹上传到：
`<user>@<gateway-host>:/mnt/afs/<user>/data_mobile/<NNNN>_<run_id>`（端口 `<ssh-port>`）。
加 `--no_upload` 可跳过；单独补传：

```powershell
python upload_success_to_remote.py --run_id 20260803_171524
```

start_from可以从以下几个阶段开始续跑：
Stages (in order):
  1. explore (adb)             
  2. process_explore      
  3. synthesize          
  4. rollout （session: 直接产出 result.json；legacy flat: strong-only / weak-to-strong）           
  5. merge (legacy flat only; 完成的轨迹中 strong step)
  6. refine_conclusion (legacy flat only)   
  7. refine_thinking (legacy flat only)
  8. upload（自动，可用 --no_upload 跳过；session 模式需在 rollout 后手动收集 result.json）
