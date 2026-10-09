> Shared eval/rollout preset: `--runtime=qwen35_thought_session`  
> Same screenshot retention as `qwen35_session` (`--last_n`).  
> Native thinking is **off**: the model writes `Thought:` in assistant content, then one XML `<tool_call>`.  
> `--enable_thinking` and `--qwen35_tool_call_mode` are ignored.

Same multi-turn loop as `qwen35_session` (query once, screenshots as tool messages, collapse old images). Response format matches the Thought: training sample instead of native `<think>` / `reasoning_content`.

### vs `qwen35_session`

| | `qwen35_session` | `qwen35_thought_session` |
|--|------------------|--------------------------|
| Reasoning | native thinking and/or `<think>` | `Thought:` paragraph in content |
| Tools | `tools=` (native) or XML in system | XML tools in the system prompt (no `tools=`) |
| `enable_thinking` | CLI flag | always `false` |

### assistant turn

```text
Thought: ...
<tool_call>
<function=mobile_use>
<parameter=action>
click
</parameter>
<parameter=coordinate>
[163, 718]
</parameter>
</function>
</tool_call>
```

### usage

AndroidWorld:

```bash
python run.py \
  --runtime qwen35_thought_session \
  --last_n 3 \
  --model_base_url http://<openai-compatible-host>/v1 \
  --model_name Qwen3.5-9B \
  --checkpoint_dir runs/qwen35-thought
```

MobileGym official eval (`--agent` maps to the same runtime; `-EnableThinking` / `-Qwen35ToolCallMode` are ignored):

```bash
cd MobileGym
./run_eval_session_mac.sh \
  -Runtime qwen35_thought_session \
  -LastN 3 \
  -ModelBaseUrl http://<openai-compatible-host>/v1 \
  -ModelName YOUR_MODEL
```

MobileGym DIY rollout already uses `--runtime`:

```bash
python run_diy_mg.py \
  --runtime qwen35_thought_session \
  --last_n 3 \
  --env_url http://127.0.0.1:4172 \
  --model_base_url http://<openai-compatible-host>/v1 \
  --model_name Qwen3.5-9B \
  --input_json prepared_tasks.json \
  --output_dir runs/qwen35-thought
```
