# Full MobileWorld pipeline in session format.
# Always: restart WSL + fresh Docker, wait until healthy, then
#   explore -> process_explore -> synthesize -> rollout
# Stops after rollout (no merge/refine/upload).
#
# Usage:
#   conda activate android_world
#   cd <OPENMOBILE_ROOT>\MobileWorld
#   .\run_full_pipeline_session.ps1
#
# Resume (still restarts WSL first):
#   .\run_full_pipeline_session.ps1 -RunId 20260813_183756 -StartFrom rollout

param(
    [string]$Runtime = "qwen35_session",
    [int]$LastN = 3,
    [string]$RunId = "",
    [string]$StartFrom = "explore",
    [int]$DockerCount = 3,
    [int]$LaunchInterval = 20,
    [int]$HealthPollSeconds = 15,
    [int]$HealthTimeoutMinutes = 12,
    [string]$WslDistro = "ubuntu",
    [string]$MwRepoInWsl = "<MOBILEWORLD_ROOT>",
    [string]$ModelBaseUrl = "https://<openai-compatible-host>/v1",
    [string]$ModelName = "gemini-3.1-pro-preview",
    [string]$ApiKey = "YOUR_API_KEY"
)

$ErrorActionPreference = "Stop"
$WorkDir = $PSScriptRoot
$StopAfter = "rollout"

$BackendHosts = @(
    for ($i = 0; $i -lt $DockerCount; $i++) {
        "http://127.0.0.1:{0}" -f (6800 + $i)
    }
)
$HostsArg = $BackendHosts -join ","

function Write-Log([string]$Message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Get-BackendHealthDetail([string]$Url) {
    $py = @"
import json, sys
import requests
url = sys.argv[1]
try:
    r = requests.get(f"{url}/health", timeout=8)
    body = r.json()
    print(json.dumps({
        "reachable": True,
        "status_code": r.status_code,
        "ok": body.get("ok"),
        "devices": body.get("devices") or [],
        "device_status": body.get("device_status") or {},
    }))
except Exception as e:
    print(json.dumps({"reachable": False, "error": str(e)}))
"@
    $raw = $py | python - $Url 2>$null
    if (-not $raw) {
        return [pscustomobject]@{ Url = $Url; Reachable = $false; Healthy = $false; Detail = "python health probe failed" }
    }
    try {
        $data = $raw | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{ Url = $Url; Reachable = $false; Healthy = $false; Detail = "bad json: $raw" }
    }
    $devices = @($data.devices)
    $healthy = ($data.reachable -eq $true -and $data.ok -eq $true -and $devices.Count -gt 0)
    $detail = if ($data.reachable) {
        "http=$($data.status_code) ok=$($data.ok) devices=$($devices -join ',')"
    } else {
        $data.error
    }
    return [pscustomobject]@{ Url = $Url; Reachable = [bool]$data.reachable; Healthy = $healthy; Detail = $detail }
}

function Initialize-Backend([string]$Url) {
    $py = @"
import sys, requests
url = sys.argv[1]
try:
    r = requests.post(f"{url}/init", json={"device": "emulator-5554"}, timeout=30)
    print(r.status_code)
except Exception as e:
    print(f"ERR:{e}")
"@
    $py | python - $Url 2>$null | Out-Null
}

function Test-WslBackendReachable([string]$Url) {
    $port = ([uri]$Url).Port
    $cmd = "curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health || echo FAIL"
    $code = wsl -d $WslDistro -e bash -lc $cmd 2>$null
    return ($code -match '^[23]')
}

function Wait-ForBackends {
    param([string[]]$Urls)

    $deadline = (Get-Date).AddMinutes($HealthTimeoutMinutes)
    Write-Log ("Waiting for backends: {0}" -f ($Urls -join ", "))

    $initDone = $false
    while ((Get-Date) -lt $deadline) {
        if (-not $initDone) {
            foreach ($u in $Urls) { Initialize-Backend $u }
            $initDone = $true
        }

        $details = @($Urls | ForEach-Object { Get-BackendHealthDetail $_ })
        $ready = @($details | Where-Object { $_.Healthy })
        if ($ready.Count -eq $Urls.Count) {
            Write-Log "All backends are healthy."
            return
        }

        $summary = ($details | ForEach-Object { "{0} => {1}" -f $_.Url, $_.Detail }) -join " | "
        Write-Log ("Ready {0}/{1}. {2}. Retry in {3}s ..." -f $ready.Count, $Urls.Count, $summary, $HealthPollSeconds)
        Start-Sleep -Seconds $HealthPollSeconds
    }

    Write-Log "Timeout diagnostics:"
    wsl -d $WslDistro -e bash -lc "cd '$MwRepoInWsl' && uv run mw env list" 2>$null
    foreach ($u in $Urls) {
        $wslOk = Test-WslBackendReachable $u
        $win = Get-BackendHealthDetail $u
        Write-Log ("  {0}: WSL curl={1}, Windows={2}" -f $u, $(if ($wslOk) { "ok" } else { "fail" }), $win.Detail)
    }
    throw "Backends not healthy within $HealthTimeoutMinutes minutes."
}

function Restart-MobileWorldDocker {
    Write-Log "Cleaning old MobileWorld containers ..."
    wsl -d $WslDistro -e bash -lc "cd '$MwRepoInWsl' && uv run mw env rm --all" 2>$null | Out-Null

    Write-Log "Shutting down WSL ..."
    wsl --shutdown
    Start-Sleep -Seconds 15

    Write-Log ("Starting {0} fresh Docker container(s) ..." -f $DockerCount)
    wsl -d $WslDistro -e bash -lc "cd '$MwRepoInWsl' && uv run mw env run --count $DockerCount --launch-interval $LaunchInterval"
    if ($LASTEXITCODE -ne 0) {
        throw "mw env run failed with exit code $LASTEXITCODE"
    }
}

function Enable-KeepAwake {
    # Prevent sleep/hibernate while this script is running.
    # Does not block Windows Update forced restart or a real shutdown.
    Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class SleepPreventer {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    $esContinuous = [uint32]"0x80000000"
    $esSystemRequired = [uint32]"0x00000001"
    $esDisplayRequired = [uint32]"0x00000002"
    [void][SleepPreventer]::SetThreadExecutionState(
        $esContinuous -bor $esSystemRequired -bor $esDisplayRequired
    )
    Write-Log "Keep-awake on (blocks sleep/hibernate for this process)."
}

Set-Location $WorkDir
Enable-KeepAwake

Write-Log ("session pipeline  runtime={0}  last_n={1}  hosts={2}" -f $Runtime, $LastN, $HostsArg)
if ($RunId) {
    Write-Log ("resume run_id={0} start_from={1} stop_after={2}" -f $RunId, $StartFrom, $StopAfter)
} else {
    Write-Log ("new run  start_from={0} stop_after={1}" -f $StartFrom, $StopAfter)
}

Restart-MobileWorldDocker
Wait-ForBackends -Urls $BackendHosts

$pyArgs = @(
    "run_full_pipeline_mw.py",
    "--hosts", $HostsArg,
    "--runtime", $Runtime,
    "--last_n", "$LastN",
    "--start_from", $StartFrom,
    "--stop_after", $StopAfter,
    "--no_upload",
    "--qwen3vl_model_base_url", $ModelBaseUrl,
    "--qwen3vl_model_name", $ModelName,
    "--qwen3vl_model_api_key", $ApiKey,
    "--openai_base_url", $ModelBaseUrl,
    "--openai_api_key", $ApiKey,
    "--openai_model", $ModelName
)
if ($RunId) {
    $pyArgs += @("--run_id", $RunId)
}

Write-Host ("=" * 72)
& python @pyArgs
$exitCode = $LASTEXITCODE
Write-Host ("=" * 72)
Write-Log ("pipeline exited with code {0}" -f $exitCode)
exit $exitCode
