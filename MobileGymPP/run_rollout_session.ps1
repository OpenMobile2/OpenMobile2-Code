# Roll hybrid GUI+App-Tools. Frontend is yours — this script does not start npm.
# Point -Port at the mock you already opened (vite default 3000).
#
#   conda activate android_world
#   cd <OPENMOBILE_ROOT>\MobileGymPP
#   .\run_rollout_session.ps1 -Port 3000 -Limit 2 -App meituan

param(
    [string]$FrontendDir = "..\..\mobilegym++\mobilegym-mock\trial_apps\mobilegym",
    [string]$EnvUrl = "",
    [int]$Port = 4172,
    [int]$Parallel = 8,
    [string]$Suite = "hybrid_tools_0831_explore",
    [string]$App = "",
    [string]$TaskIds = "",
    [int]$Limit = 0,
    [string]$InteractionMode = "hybrid",
    [string]$Runtime = "qwen35_session",
    [int]$LastN = 3,
    [string]$EnableThinking = "true",
    [string]$ModelBaseUrl = "https://<openai-compatible-host>/v1",
    [string]$ModelName = "gemini-3.1-pro-preview",
    [string]$ApiKey = "YOUR_API_KEY",
    [string]$PythonCondaEnv = "android_world",
    [string]$PythonExe = "",
    [string]$RunsDir = "runs_rollout\gemini",
    [switch]$List,
    [switch]$NoHeadless,
    [switch]$NoResume,
    [switch]$LiveConsole,
    [double]$StartDelay = 0
)

$ErrorActionPreference = "Stop"
if (-not [System.IO.Path]::IsPathRooted($FrontendDir)) {
    $FrontendDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $FrontendDir))
}
$WorkDir = $PSScriptRoot
$exitCode = 1

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
    if (-not ("SleepPreventerMgMcp" -as [type])) {
        Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class SleepPreventerMgMcp {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    }
    $esContinuous = [uint32]"0x80000000"
    $esSystemRequired = [uint32]"0x00000001"
    $esDisplayRequired = [uint32]"0x00000002"
    [void][SleepPreventerMgMcp]::SetThreadExecutionState(
        $esContinuous -bor $esSystemRequired -bor $esDisplayRequired
    )
}

function Get-EvalPython {
    if ($PythonExe -and (Test-Path $PythonExe)) { return $PythonExe }
    $condaPy = Join-Path $env:USERPROFILE "miniconda3\envs\$PythonCondaEnv\python.exe"
    if (Test-Path $condaPy) { return $condaPy }
    throw "python not found in conda env '$PythonCondaEnv'. Expected: $condaPy"
}

function Get-SafeRunName([string]$Name) {
    $safe = $Name.Trim().Replace(".", "")
    $safe = [regex]::Replace($safe, '[<>:"/\\|?*\s]+', "_")
    $safe = [regex]::Replace($safe, "_+", "_").Trim("._")
    if (-not $safe) { $safe = "run" }
    return $safe
}

Set-Location $WorkDir
Enable-KeepAwake
$evalPython = Get-EvalPython

if ($List) {
    $listArgs = @("run_rollout.py", "--list", "--suite", $Suite, "--mock-root", $FrontendDir)
    if ($App) { $listArgs += @("--app", $App) }
    & $evalPython @listArgs
    exit $LASTEXITCODE
}

$runsRoot = if ([System.IO.Path]::IsPathRooted($RunsDir)) {
    [System.IO.Path]::GetFullPath($RunsDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $WorkDir $RunsDir))
}
$runName = "{0}_{1}_{2}" -f (Get-SafeRunName $Suite), $InteractionMode, (Get-SafeRunName $ModelName)
$outputDir = Join-Path $runsRoot $runName

$pyArgs = @(
    "run_rollout.py",
    "--mock-root", $FrontendDir,
    "--suite", $Suite,
    "--interaction-mode", $InteractionMode,
    "--runtime", $Runtime,
    "--last-n", "$LastN",
    "--env-url", $EnvUrl,
    "--model-base-url", $ModelBaseUrl,
    "--model-name", $ModelName,
    "--model-api-key", $ApiKey,
    "--enable-thinking", $EnableThinking,
    "--parallel", "$Parallel",
    "--output-dir", $outputDir
)
if ($LiveConsole) { $NoHeadless = $true }
if (-not $NoHeadless) { $pyArgs += "--headless" }
if ($NoResume) { $pyArgs += "--no-resume" }
if ($LiveConsole) { $pyArgs += "--live-console" }
if ($StartDelay -gt 0) { $pyArgs += @("--start-delay", "$StartDelay") }
if ($App) { $pyArgs += @("--app", $App) }
if ($TaskIds) { $pyArgs += @("--task-ids", $TaskIds) }
if ($Limit -gt 0) { $pyArgs += @("--limit", "$Limit") }

Write-Host ("=" * 72)
Write-Log ("env-url={0} suite={1} parallel={2}" -f $EnvUrl, $Suite, $Parallel)
Write-Log ("python {0}" -f ($pyArgs -join " "))
& $evalPython @pyArgs
$exitCode = $LASTEXITCODE
if ($exitCode -eq 0 -or (Test-Path (Join-Path $outputDir "episodes"))) {
    Write-Log "Collecting traces.jsonl ..."
    & $evalPython "collect_traces.py" "--run-dir" $outputDir "--require-app-tool"
}
Write-Log ("rollout exited with code {0}" -f $exitCode)
exit $(if ($null -eq $exitCode) { 1 } else { $exitCode })
