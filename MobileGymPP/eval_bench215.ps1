# One bench215 launcher. Pick the model with -Runtime, -ModelName, and -ModelBaseUrl.
# Does not start npm. Point -Port at an already-running MobileGym++ preview.
#
#   conda activate android_world
#   cd <OPENMOBILE_ROOT>\MobileGymPP
#   .\eval_bench215.ps1 -List
#   .\eval_bench215.ps1 -Runtime qwen35_thought_session -Port 4173 -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName YOUR_MODEL
#   .\eval_bench215.ps1 -Runtime qwen3vl -Mode gui_only -Port 4173 -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName OpenMobile-8B
#   .\eval_bench215.ps1 -Runtime venus -Port 4173 -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName UI-Venus-1.5-8B
#   .\eval_bench215.ps1 -Runtime gui_owl -Port 4173 -ModelBaseUrl http://<openai-compatible-host>/v1 -ModelName GUI-Owl-1.5-8B
#
# -Mode both runs hybrid, then gui_only. last_n is locked by runtime
# (thought_session 3, gui_owl 5, mai_ui 3, venus 1, qwen3vl 1) unless you pass -LastN
# for thought_session.

param(
    [string]$FrontendDir = "..\..\mobilegym++\mobilegym-mock\trial_apps\mobilegym",
    [string]$EnvUrl = "",
    [int]$Port = 4173,
    [ValidateSet("both", "hybrid", "gui_only")]
    [string]$Mode = "both",
    [ValidateSet("all", "main", "secretary")]
    [string]$Pack = "all",
    [int]$Parallel = 8,
    [int]$Limit = 0,
    [string]$TaskIds = "",
    [ValidateSet("qwen35_thought_session", "gui_owl", "mai_ui", "venus", "qwen3vl")]
    [string]$Runtime = "qwen35_thought_session",
    [int]$LastN = 0,
    [string]$ModelBaseUrl = "http://<openai-compatible-host>/v1",
    [string]$ModelName = "",
    [string]$ApiKey = "EMPTY",
    [string]$PythonCondaEnv = "android_world",
    [string]$PythonExe = "",
    [string]$OutputDir = "",
    [switch]$List,
    [switch]$NoHeadless,
    [switch]$NoResume,
    [double]$StartDelay = 0
)

$ErrorActionPreference = "Stop"
if (-not [System.IO.Path]::IsPathRooted($FrontendDir)) {
    $FrontendDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $FrontendDir))
}
if ($OutputDir -and -not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $OutputDir))
}
$WorkDir = $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:MOBILEGYM_MOCK = $FrontendDir

if (-not $EnvUrl) {
    $EnvUrl = "http://127.0.0.1:$Port"
}

function Write-Log([string]$Message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Enable-KeepAwake {
    if (-not ("SleepPreventerBench215" -as [type])) {
        Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class SleepPreventerBench215 {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    }
    $esContinuous = [uint32]"0x80000000"
    $esSystemRequired = [uint32]"0x00000001"
    $esDisplayRequired = [uint32]"0x00000002"
    [void][SleepPreventerBench215]::SetThreadExecutionState(
        $esContinuous -bor $esSystemRequired -bor $esDisplayRequired
    )
}

function Get-EvalPython {
    if ($PythonExe -and (Test-Path $PythonExe)) { return $PythonExe }
    $condaPy = Join-Path $env:USERPROFILE "miniconda3\envs\$PythonCondaEnv\python.exe"
    if (Test-Path $condaPy) { return $condaPy }
    throw "python not found in conda env '$PythonCondaEnv'. Expected: $condaPy"
}

Set-Location $WorkDir
$evalPython = Get-EvalPython

if ($List) {
    & $evalPython "eval_bench215.py" "--list" "--mock-root" $FrontendDir "--runtime" $Runtime
    exit $LASTEXITCODE
}

if (-not $ModelName) {
    throw "pass -ModelName -ModelBaseUrl -Runtime"
}

Enable-KeepAwake

$pyArgs = @(
    "eval_bench215.py",
    "--mock-root", $FrontendDir,
    "--mode", $Mode,
    "--pack", $Pack,
    "--env-url", $EnvUrl,
    "--model-base-url", $ModelBaseUrl,
    "--model-name", $ModelName,
    "--model-api-key", $ApiKey,
    "--parallel", "$Parallel",
    "--runtime", $Runtime
)
if ($LastN -gt 0) { $pyArgs += @("--last-n", "$LastN") }
if ($OutputDir) { $pyArgs += @("--output-dir", $OutputDir) }
if ($Limit -gt 0) { $pyArgs += @("--limit", "$Limit") }
if ($TaskIds) { $pyArgs += @("--task-ids", $TaskIds) }
if ($NoHeadless) { $pyArgs += "--no-headless" }
if ($NoResume) { $pyArgs += "--no-resume" }
if ($StartDelay -gt 0) { $pyArgs += @("--start-delay", "$StartDelay") }

Write-Host ("=" * 72)
Write-Log ("bench215 runtime={0} mode={1} env-url={2} parallel={3}" -f $Runtime, $Mode, $EnvUrl, $Parallel)
Write-Log ("python {0}" -f ($pyArgs -join " "))
& $evalPython @pyArgs
$exitCode = $LASTEXITCODE
Write-Log ("eval_bench215 exited with code {0}" -f $exitCode)
exit $(if ($null -eq $exitCode) { 1 } else { $exitCode })
