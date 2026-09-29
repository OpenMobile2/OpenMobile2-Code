# Two conda envs, do not mix:
#   mobilegym     -> npm frontend (<MOBILEGYM_FRONTEND>)
#   android_world -> python eval  (this repo)
#
# One official-scoring launcher. Pick the model with -Runtime, -ModelName, -ModelBaseUrl.
#   .\run_eval_session.ps1 -Runtime qwen35_thought_session -LastN 3 -ModelName YOUR_MODEL -ModelBaseUrl http://<openai-compatible-host>/v1
#   .\run_eval_session.ps1 -Runtime qwen3vl -ModelName OpenMobile-8B -ModelBaseUrl http://<openai-compatible-host>/v1
#   .\run_eval_session.ps1 -Runtime venus -ModelName UI-Venus-1.5-8B -ModelBaseUrl http://<openai-compatible-host>/v1
#   .\run_eval_session.ps1 -Runtime gui_owl -ModelName GUI-Owl-1.5-8B -ModelBaseUrl http://<openai-compatible-host>/v1
#
# venus and qwen3vl lock last_n=1. thought_session defaults to 3.
# qwen3vl writes runs_main/qwen3vl/<ModelName> so it does not resume the old
# uitars folder runs_main/OpenMobile-8B. Other runtimes write
# runs_main/<ModelName with dots removed>. An existing meta.json resumes.
#
#   .\run_eval_session.ps1 -Resume <OPENMOBILE_ROOT>\MobileGym\runs_main\YOUR_MODEL -Runtime qwen35_thought_session -ModelName YOUR_MODEL
#   .\run_eval_session.ps1 -Split "" -FilterDifficulty L1 -Runtime qwen35_thought_session -ModelName YOUR_MODEL

param(
    [string]$FrontendDir = "..\..\mobilegym\mobilegym",
    [string]$EnvUrl = "http://127.0.0.1:4172",
    [int]$Port = 4172,
    [int]$Parallel = 8,
    [int]$Processes = 1,
    [string]$Split = "test",
    [string]$TaskId = "",
    [string]$Suite = "",
    [string]$FilterDifficulty = "",
    [ValidateSet("qwen35_thought_session", "qwen35_session", "qwen3vl", "venus", "gui_owl")]
    [string]$Runtime = "qwen35_thought_session",
    [int]$LastN = 3,
    [string]$EnableThinking = "true",
    [string]$RequireThinkTags = "false",
    [string]$Qwen35ToolCallMode = "native",
    [string]$ModelBaseUrl = "http://<openai-compatible-host>/v1",
    [string]$ModelName = "",
    [string]$ApiKey = "EMPTY",
    [string]$NodeCondaEnv = "mobilegym",
    [string]$PythonCondaEnv = "android_world",
    [string]$PythonExe = "",
    [int]$HealthTimeoutMinutes = 8,
    [string]$RunsDir = "runs_main",
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
if (-not [System.IO.Path]::IsPathRooted($FrontendDir)) {
    $FrontendDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $FrontendDir))
}
if (-not $PSBoundParameters.ContainsKey("LastN")) {
    if ($Runtime -eq "venus" -or $Runtime -eq "qwen3vl") { $LastN = 1 }
    elseif ($Runtime -eq "qwen35_thought_session") { $LastN = 3 }
}
# Flat ReAct must not resume the old uitars folder runs_main/OpenMobile-8B.
if (-not $PSBoundParameters.ContainsKey("RunsDir") -and $Runtime -eq "qwen3vl") {
    $RunsDir = "runs_main\qwen3vl"
}
if ($Runtime -eq "qwen35_thought_session") {
    $EnableThinking = "false"
    $Qwen35ToolCallMode = "xml"
}
if (-not $ModelName) {
    throw "pass -ModelName -ModelBaseUrl -Runtime"
}
$WorkDir = $PSScriptRoot
$previewProc = $null
$startedPreview = $false
$exitCode = 1

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

function Write-Log([string]$Message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Enable-KeepAwake {
    if (-not ("SleepPreventerMgEval" -as [type])) {
        Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class SleepPreventerMgEval {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    }
    $esContinuous = [uint32]"0x80000000"
    $esSystemRequired = [uint32]"0x00000001"
    $esDisplayRequired = [uint32]"0x00000002"
    [void][SleepPreventerMgEval]::SetThreadExecutionState(
        $esContinuous -bor $esSystemRequired -bor $esDisplayRequired
    )
    Write-Log "Keep-awake on."
}

function Test-FrontendUp([string]$Url) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
    } catch {
        return $false
    }
}

function Get-NpmCmd {
    $condaNpm = Join-Path $env:USERPROFILE "miniconda3\envs\$NodeCondaEnv\npm.cmd"
    if (Test-Path $condaNpm) { return $condaNpm }
    throw "npm not found in conda env '$NodeCondaEnv'. Expected: $condaNpm"
}

function Get-EvalPython {
    if ($PythonExe -and (Test-Path $PythonExe)) { return $PythonExe }
    $condaPy = Join-Path $env:USERPROFILE "miniconda3\envs\$PythonCondaEnv\python.exe"
    if (Test-Path $condaPy) { return $condaPy }
    throw "python not found in conda env '$PythonCondaEnv'. Expected: $condaPy"
}

function Invoke-Npm {
    param([string[]]$NpmArgs)
    $npm = Get-NpmCmd
    Write-Log ("npm ({0}) {1}" -f $npm, ($NpmArgs -join " "))
    & $npm @NpmArgs
    if ($LASTEXITCODE -ne 0) {
        throw "npm failed with exit code $LASTEXITCODE : $($NpmArgs -join ' ')"
    }
}

function Start-MobileGymFrontend {
    param([string]$Dir, [int]$ListenPort, [string]$Url)

    if (-not (Test-Path $Dir)) {
        throw "Frontend dir not found: $Dir"
    }
    if (Test-FrontendUp $Url) {
        Write-Log ("Frontend already up at {0}, skip deploy." -f $Url)
        return
    }

    Push-Location $Dir
    try {
        if (-not (Test-Path (Join-Path $Dir "node_modules"))) {
            Write-Log "npm install ..."
            Invoke-Npm @("install")
        }
        if (-not (Test-Path (Join-Path $Dir "dist\index.html"))) {
            Write-Log "npm run build ..."
            Invoke-Npm @("run", "build")
        }
        $npm = Get-NpmCmd
        Write-Log ("Starting preview on port {0} ..." -f $ListenPort)
        $script:previewProc = Start-Process -FilePath $npm -ArgumentList @(
            "run", "preview", "--", "--host", "127.0.0.1", "--port", "$ListenPort"
        ) -WorkingDirectory $Dir -PassThru -WindowStyle Hidden
        $script:startedPreview = $true
    } finally {
        Pop-Location
    }

    $deadline = (Get-Date).AddMinutes($HealthTimeoutMinutes)
    while ((Get-Date) -lt $deadline) {
        if (Test-FrontendUp $Url) {
            Write-Log ("Frontend healthy: {0}" -f $Url)
            return
        }
        if ($script:previewProc -and $script:previewProc.HasExited) {
            throw "npm preview exited early (code $($script:previewProc.ExitCode))."
        }
        Write-Log "Waiting for frontend ..."
        Start-Sleep -Seconds 3
    }
    throw "Frontend not reachable at $Url within $HealthTimeoutMinutes minutes."
}

function Stop-MobileGymFrontend {
    if (-not $startedPreview -or $null -eq $previewProc) { return }
    Write-Log "Stopping frontend preview ..."
    try {
        Start-Process -FilePath "taskkill.exe" -ArgumentList "/PID", "$($previewProc.Id)", "/T", "/F" -Wait -WindowStyle Hidden | Out-Null
    } catch {
        Write-Log ("taskkill failed: {0}" -f $_.Exception.Message)
    }
}

function Get-SafeRunName([string]$Name) {
    $safe = $Name.Trim().Replace(".", "")
    $safe = [regex]::Replace($safe, '[<>:"/\\|?*\s]+', "_")
    $safe = [regex]::Replace($safe, "_+", "_").Trim("._")
    if (-not $safe) { $safe = "run" }
    return $safe
}

function Get-CommonEvalArgs {
    $pyArgs = @(
        "--env-url", $EnvUrl,
        "--agent", $Runtime,
        "--parallel", "$Parallel",
        "--processes", "$Processes",
        "--headless",
        "--last_n", "$LastN",
        "--enable_thinking", $EnableThinking,
        "--require_think_tags", $RequireThinkTags,
        "--qwen35_tool_call_mode", $Qwen35ToolCallMode,
        "--model-base-url", $ModelBaseUrl,
        "--model-name", $ModelName,
        "--model-api-key", $ApiKey,
        "--runs-dir", $runsRoot
    )
    if ($TaskId) {
        $pyArgs += @("--task-id", $TaskId)
    } elseif ($Suite) {
        $pyArgs += @("--suite", $Suite)
    } elseif ($Split) {
        $pyArgs += @("--split", $Split)
    } elseif ($FilterDifficulty) {
        $pyArgs += @("--filter-difficulty", $FilterDifficulty)
    }
    return ,$pyArgs
}

Set-Location $WorkDir
Enable-KeepAwake
$env:MOBILEGYM_FRONTEND = $FrontendDir
$evalPython = Get-EvalPython

Write-Log ("npm from conda env {0}; python from conda env {1}" -f $NodeCondaEnv, $PythonCondaEnv)
Write-Log ("python={0}" -f $evalPython)
& $evalPython -c "import requests, playwright, openai, PIL"
if ($LASTEXITCODE -ne 0) {
    throw "android_world env is missing packages. Run: conda activate android_world; pip install requests -r `"$FrontendDir\bench_env\requirements.txt`"; playwright install chromium"
}

Write-Log ("MobileGym EVAL  split={0} agent={1} last_n={2} thinking={3} require_think_tags={4} mode={5}" -f `
    $(if ($TaskId) { "task:$TaskId" } elseif ($Suite) { "suite:$Suite" } elseif ($Split) { $Split } elseif ($FilterDifficulty) { "difficulty:$FilterDifficulty" } else { "ALL" }), `
    $Runtime, $LastN, $EnableThinking, $RequireThinkTags, $Qwen35ToolCallMode)
Write-Log ("frontend={0} env={1} parallel={2} processes={3}" -f $FrontendDir, $EnvUrl, $Parallel, $Processes)

try {
    Start-MobileGymFrontend -Dir $FrontendDir -ListenPort $Port -Url $EnvUrl

    $runsRoot = if ([System.IO.Path]::IsPathRooted($RunsDir)) {
        [System.IO.Path]::GetFullPath($RunsDir)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $WorkDir $RunsDir))
    }
    $fixedRunDir = Join-Path $runsRoot (Get-SafeRunName $ModelName)
    $common = Get-CommonEvalArgs

    if ($Resume) {
        $target = $Resume
        if (-not [System.IO.Path]::IsPathRooted($target)) {
            $target = [System.IO.Path]::GetFullPath((Join-Path $WorkDir $target))
        }
    } elseif (Test-Path (Join-Path $fixedRunDir "meta.json")) {
        $target = $fixedRunDir
    } else {
        $target = $null
    }

    Write-Host ("=" * 72)
    if ($target) {
        Write-Log ("OpenMobile resume (append into): {0}" -f $target)
        $pyArgs = @("resume_eval_mg.py", "--run-dir", $target) + $common
        Write-Log ("python {0}" -f ($pyArgs -join " "))
        & $evalPython @pyArgs
        $exitCode = $LASTEXITCODE
    } else {
        Write-Log ("Fresh run dir: {0}" -f $fixedRunDir)
        $pyArgs = @("run_eval_mg.py") + $common + @("--run-dir", $fixedRunDir)
        Write-Log ("python {0}" -f ($pyArgs -join " "))
        & $evalPython @pyArgs
        $exitCode = $LASTEXITCODE
    }
    Write-Host ("=" * 72)
    Write-Log ("eval exited with code {0}" -f $exitCode)
} finally {
    Stop-MobileGymFrontend
}

exit $(if ($null -eq $exitCode) { 1 } else { $exitCode })
