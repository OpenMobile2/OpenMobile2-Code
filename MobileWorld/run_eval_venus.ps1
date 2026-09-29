# MobileWorld eval for UI-Venus-1.5.
# Runtime = AndroidWorld `--runtime venus` (mirrors MobileGym official
# bench_env.agent.venus.VenusAgent: <think>/<action>/<conclusion>, HISTORY_N=1).
#
#   conda activate android_world
#   cd <OPENMOBILE_ROOT>\MobileWorld
#   .\run_eval_venus.ps1
#   .\run_eval_venus.ps1 -ModelBaseUrl http://<openai-compatible-host>/v1
#   .\run_eval_venus.ps1 -Phase smoke -Tasks SOME_TASK

param(
    [ValidateSet("gui", "smoke")]
    [string]$Phase = "gui",
    [string]$Hosts = "http://127.0.0.1:6800",
    [string]$ModelBaseUrl = "http://<openai-compatible-host>/v1",
    [string]$ModelName = "UI-Venus-1.5-8B",
    [string]$ApiKey = "EMPTY",
    [string]$OutputDir = "eval-gui/UI-Venus-1.5-8B/gui-only",
    [string]$Tasks = "ALL",
    [int]$LastN = 1,
    [int]$MaxNSteps = 50,
    [string]$PythonExe = ""
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
$argsList = @(
    "parallel_eval_mw.py",
    "--hosts", $Hosts,
    "--runtime", "venus",
    "--last_n", "$LastN",
    "--max_n_steps", "$MaxNSteps",
    "--qwen3vl_model_base_url", $ModelBaseUrl,
    "--qwen3vl_model_name", $ModelName,
    "--qwen3vl_model_api_key", $ApiKey,
    "--output_dir", $OutputDir
)

if ($Phase -eq "smoke") {
    if ($Tasks -eq "ALL") {
        Write-Host "smoke phase needs -Tasks <one_or_more_task_ids>"
        exit 2
    }
    $argsList += @("--tasks", $Tasks)
} else {
    $argsList += @("--tasks", $Tasks)
}

Write-Host ("Launch Venus MW {0}: {1} {2}" -f $Phase, $py, ($argsList -join " "))
& $py @argsList
exit $LASTEXITCODE
