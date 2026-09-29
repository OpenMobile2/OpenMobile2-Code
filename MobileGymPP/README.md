# MobileGymPP

专门 roll **hybrid GUI + App Tools** 轨迹。bench215 评测用
**qwen35_thought_session**（`Thought:` + XML `<tool_call>`，不发 `tools=`）。

轨迹合成代码仍保留 `qwen35_session`：system 只发一次、题面只发一次、之后回截图。
`tools=` 整场锁死 `[mobile_use]`；打开新 App 时，该步 user/观察文本里带上
这个 App 的 MCP schema，不改 system 前缀。

默认任务集是 mock 里的 `hybrid_tools_custom_v5`（2040 条，无语义 verifier）。

## mock 里 GUI-only 和 GUI+tools 差在哪

不是两套 agent。`HybridAgentRuntime` 用 `--interaction-mode` 开关：

| | `gui_only` | `hybrid` |
|---|---|---|
| 发现 | 只看前台 App，不调 `listTools` | `listTools({app})` |
| 模型可见 | 只有 `mobile_use` | `tools=` 仍只有 `mobile_use`；切 App 当步 user 文本带该 App tools |
| 执行 | `env.step`（点/滑/打字） | GUI 同上；其它名字走 `callTool` |
| 桌面 / 最近任务 / 控制中心 | 只有 `mobile_use` | 同样只有 `mobile_use`，必须先打开 App |

`candidate_tools` 只是任务元数据，**不会**进 prompt。没有单独的 MCP 进程；
页面上的 `window.__MOBILE_GYM_TOOLS__` 就是工具源。

## 这边怎么接到 session

官方 `../MobileGym/` 的 session 已经是 native function calling：

```text
tools = [mobile_use]
```

hybrid session 只多做两件事（见 `session_tools.py` / `session_runtime.py`）：

```text
整场：tools = [mobile_use]
刚切到新 App 的那一步：user/观察文本写入 listTools(foreground)
非 mobile_use → callTool，下一轮观察带 JSON + 截图
```

`Qwen35Session` 上的钩子：`set_turn_tools`（整场 `[mobile_use]`）、
`set_next_user_text`（切 App 时写入 schema）、`set_next_observation_text`（tool JSON），
以及 `name != mobile_use` 时返回 `action_type=mcp_tool`（不走 GUI transform）。
GUI-only 评测路径不变。

`--runtime hybrid_xml` 才走 mock 原来的 `HybridXmlAgent`（每步重写 system，
不是 session 格式）。

## 环境

两套 conda，不要混：

| env | 用途 |
|---|---|
| `mobilegym` | `npm run preview` 起模拟器 |
| `android_world` | 本目录的 Python |

前端必须是 **mobilegym-mock**（custom_v5 只在这里）：

```text
<MOBILEGYM_MOCK_ROOT>\trial_apps\mobilegym
```

不要指到 `<MOBILEGYM_FRONTEND>`。

```bash
conda activate mobilegym
cd <MOBILEGYM_MOCK_ROOT>\trial_apps\mobilegym
npm run build && npm run preview -- --host 127.0.0.1 --port 4172
```

```bash
conda activate android_world
pip install -r <MOBILEGYM_MOCK_ROOT>\trial_apps\mobilegym\bench_env\requirements.txt
playwright install chromium
```

## 评测 bench215

题单来自 mock `bench215/official.json`（214 + 秘书）。一个入口 `eval_bench215.py`，用 `--runtime` 换模型格式：`qwen35_thought_session`（`Thought:` + XML，不发 `tools=`）、`qwen3vl`、`venus`、`gui_owl`、`mai_ui`。

秘书单独跑。先自己起模拟器。

```powershell
conda activate mobilegym
cd <MOBILEGYM_MOCK_ROOT>\trial_apps\mobilegym
npm run preview -- --host 127.0.0.1 --port 4173
```

```powershell
conda activate android_world
cd <OPENMOBILE_ROOT>\MobileGymPP

python eval_bench215.py --list

python eval_bench215.py `
  --runtime qwen35_thought_session `
  --env-url http://127.0.0.1:4173 `
  --model-base-url http://<openai-compatible-host>/v1 `
  --model-name YOUR_MODEL `
  --model-api-key EMPTY `
  --parallel 5

.\eval_bench215.ps1 -Runtime qwen3vl -Mode gui_only -Port 4173 -ModelName OpenMobile-8B
.\eval_bench215.ps1 -Runtime venus -Port 4173 -ModelName UI-Venus-1.5-8B
```

`--runtime` 取 `qwen35_thought_session`、`qwen3vl`、`venus`、`gui_owl`、`mai_ui`。默认 `--mode both`：先 hybrid，再 gui_only。只要其中一种时加 `--mode hybrid` 或 `gui_only`。

结果：

```text
eval_runs/<model>/bench215-mock/
  hybrid/summary.json
  gui_only/summary.json
  comparison.json
```

## 怎么 roll

```powershell
conda activate android_world
cd <OPENMOBILE_ROOT>\MobileGymPP

.\run_rollout_session.ps1 -List
.\run_rollout_session.ps1 -List -App meituan
.\run_rollout_session.ps1 -Limit 2 -App meituan
.\run_rollout_session.ps1 -Parallel 8
.\run_rollout_session.ps1 -Suite hybrid_tools_custom_v4
```

默认：`--runtime qwen35_thought_session`、`--last-n 3`。模型地址和名字换成你的服务，例如 `http://<openai-compatible-host>/v1` / `YOUR_MODEL`。

Headed 演示（协议仍是 session，不要改去 `hybrid_benchmark`）：不要加 `--headless`，加上 `--live-console` 会多开一个思考/动作侧栏。`--live-console` 会自动取消 headless。

```powershell
python run_rollout.py `
  --mock-root <MOBILEGYM_MOCK_ROOT>\trial_apps\mobilegym `
  --suite hybrid_demo_crossapp `
  --task-ids HTV1SecretaryWangHangzhouPrep `
  --runtime qwen35_thought_session `
  --last-n 3 `
  --live-console `
  --start-delay 8 `
  --env-url http://localhost:3000 `
  --model-base-url https://<openai-compatible-host>/v1 `
  --model-name gemini-3.1-pro-preview `
  --model-api-key $env:OPENAI_API_KEY `
  --output-dir runs\secretary-session-demo
```

GLM-5.3-Flash（模型名带 `glm` / `z-ai/` 时自动切 backend）：thinking 走 `thinking.type=enabled` + `--reasoning-effort`，App Tool 进 `tools=`，不要再发 Qwen 的 `chat_template_kwargs.enable_thinking`。`--reasoning-effort low|high|max` 会传到 GLM；思考关不掉，`--enable-thinking false` 只会把 effort 降成 `low`。

```powershell
python run_rollout.py `
  --mock-root <MOBILEGYM_MOCK_ROOT>\trial_apps\mobilegym `
  --suite hybrid_demo_crossapp `
  --task-ids HTV1SecretaryWangHangzhouPrep `
  --runtime qwen35_thought_session `
  --last-n 3 `
  --live-console `
  --reasoning-effort max `
  --env-url http://localhost:4172 `
  --model-base-url https://api.qnaigc.com/v1 `
  --model-name z-ai/glm-5.3-flash `
  --model-api-key $env:OPENAI_API_KEY `
  --output-dir runs\secretary-session-glm
```

```bash
cd <OPENMOBILE_ROOT>\MobileGymPP

python run_rollout.py `
  --env-url http://127.0.0.1:4172 `
  --model-base-url https://<openai-compatible-host>/v1 `
  --model-name "gemini-3.1-pro-preview" `
  --model-api-key "YOUR_API_KEY" `
  --runtime qwen35_thought_session `
  --last-n 3 `
  --suite hybrid_tools_0831_explore `
  --interaction-mode hybrid `
  --parallel 8 --headless `
  --output-dir <OPENMOBILE_ROOT>\MobileGymPP\runs_rollout\gemini\hybrid_tools_0831_explore
```

```bash
# gui-only
cd <OPENMOBILE_ROOT>\MobileGymPP

python run_rollout.py `
  --env-url http://127.0.0.1:4172 `
  --model-base-url http://<openai-compatible-host>/v1 `
  --model-name YOUR_MODEL `
  --model-api-key EMPTY `
  --runtime qwen35_thought_session `
  --last-n 3 `
  --suite hybrid_tools_0815_explore `
  --interaction-mode gui_only `
  --parallel 8 --headless `
  --output-dir <OPENMOBILE_ROOT>\MobileGymPP\eval_runs\YOUR_MODEL\gui_only_0815_explore
```

## 产物

```text
runs/<suite>_<mode>_<model>/
  results.jsonl
  summary.json
  episodes/<task>/session.json   # 和官方 eval 同形的 messages + images
  episodes/<task>/trace.json
  episodes/<task>/index.html
  traces.jsonl
```

**v5 的 SR 不能当「做对了」。** 筛数据看有没有调 App Tool，或改用 `hybrid_tools_custom_v4`。
