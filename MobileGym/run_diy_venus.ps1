# MobileGym DIY / rollout for UI-Venus-1.5 via AndroidWorld agent_factory
# (`--runtime venus`). Same stack as the qwen3vl ReAct path
# (mg_env + create_gui_agent), but Venus native <think>/<action>/<conclusion>.
#
# Frontend must already be up (preview / nginx). This does NOT score SR —
# for official bench scoring use .\run_eval_session.ps1 -Runtime venus instead.
#
#   conda activate android_world
#   cd <OPENMOBILE_ROOT>\MobileGym
#   .\run_diy_venus.ps1 `
#     -InputJson prepared_tasks.json `
#     -EnvUrl http://127.0.0.1:4172 `
#     -ModelBaseUrl http://<openai-compatible-host>/v1

param(
    [Parameter(Mandatory = $true)]
    [string]$InputJson,
    [string]$OutputDir = "runs\UI-Venus-1.5-8B\diy",
    [string]$EnvUrl = "http://127.0.0.1:4172",
    [int]$Parallel = 1,
    [int]$MaxNSteps = 30,
    [int]$LastN = 1,
    [string]$ModelBaseUrl = "http://<openai-compatible-host>/v1",
    [string]$ModelName = "UI-Venus-1.5-8B",
    [string]$ApiKey = "EMPTY",
    [string]$PythonExe = "",
    [switch]$NoHeadless
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-EvalPython {
    if ($PythonExe -and (Test-Path $PythonExe)) { return $PythonExe }
    $condaPy = Join-Path $env:USERPROFILE "miniconda3\envs\android_world\python.exe"
    if (Test-Path $condaPy) { return $condaPy }
    return "python"
}

$py = Get-EvalPython

if ($Parallel -gt 1) {
    $argsList = @(
        "parallel_rollout_mg.py",
        "--env_url", $EnvUrl,
        "--parallel", "$Parallel",
        "--input_json", $InputJson,
        "--output_dir", $OutputDir,
        "--runtime", "venus",
        "--last_n", "$LastN",
        "--max_n_steps", "$MaxNSteps",
        "--qwen3vl_model_base_url", $ModelBaseUrl,
        "--qwen3vl_model_name", $ModelName,
        "--qwen3vl_model_api_key", $ApiKey
    )
    if ($NoHeadless) { $argsList += "--no_headless" }
} else {
    $argsList = @(
        "run_diy_mg.py",
        "--env_url=$EnvUrl",
        "--input_json=$InputJson",
        "--output_dir=$OutputDir",
        "--runtime=venus",
        "--last_n=$LastN",
        "--max_n_steps=$MaxNSteps",
        "--qwen3vl_model_base_url=$ModelBaseUrl",
        "--qwen3vl_model_name=$ModelName",
        "--qwen3vl_model_api_key=$ApiKey",
        "--headless=$(if ($NoHeadless) { 'false' } else { 'true' })"
    )
}

Write-Host ("Launch Venus MG DIY: {0} {1}" -f $py, ($argsList -join " "))
& $py @argsList
exit $LASTEXITCODE
