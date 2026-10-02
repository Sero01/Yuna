# Starts Yuna at login: the backend first, then the desktop app (pet mode) once the
# backend is listening, since the app only connects once on launch.
# Run hidden at login by Yuna.vbs in the user's Startup folder.

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$app = Join-Path $env:LOCALAPPDATA "Programs\open-llm-vtuber\open-llm-vtuber-electron.exe"
$port = 12393
$waitSeconds = 300
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

function Test-Port {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

if (-not (Test-Port)) {
    $env:PYTHONIOENCODING = "utf-8"
    Start-Process -FilePath $python -ArgumentList "run_server.py" `
        -WorkingDirectory $repo -NoNewWindow `
        -RedirectStandardOutput (Join-Path $logDir "startup-server.out.log") `
        -RedirectStandardError (Join-Path $logDir "startup-server.err.log")
    $deadline = (Get-Date).AddSeconds($waitSeconds)
    while (-not (Test-Port) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
    }
}

if ((Test-Path $app) -and -not (Get-Process "open-llm-vtuber-electron" -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $app
}
