# Start Milvus (if docker is available), then API + web, ingest doc.md, keep server in foreground.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$port = if ($env:PORT) { $env:PORT } else { "8001" }

$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($docker) {
    Write-Host "Starting Milvus stack (docker compose)..."
    docker compose up -d
    Write-Host "Waiting for Milvus port 19530..."
    $deadline = (Get-Date).AddMinutes(4)
    $open = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect("localhost", 19530)
            $tcp.Close()
            $open = $true
            Write-Host "Milvus port is open."
            break
        } catch {
            Start-Sleep -Seconds 3
        }
    }
    if (-not $open) {
        Write-Host "Warning: port 19530 did not open in time. Ingest may fail until Milvus is ready."
    }
} else {
    Write-Host "Docker not in PATH — start Milvus yourself (e.g. Docker Desktop + docker compose up -d in rag_new)."
}

$env:PYTHONPATH = $Root
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }

Write-Host "Starting uvicorn (background process)..."
$proc = Start-Process -FilePath "python" -ArgumentList @("serve.py") -WorkingDirectory $Root -PassThru -WindowStyle Hidden

Write-Host "Waiting for http://127.0.0.1:$port/health ..."
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3 | Out-Null
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $ready) {
    Write-Host "Server did not become ready. Stopping background python."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "Ingesting knowledge (POST /ingest/doc) — may take several minutes on first run (embedding model)..."
try {
    $ing = Invoke-RestMethod -Uri "http://127.0.0.1:$port/ingest/doc" -Method Post -ContentType "application/json" -Body "{}" -TimeoutSec 1200
    Write-Host ($ing | ConvertTo-Json -Depth 6)
} catch {
    Write-Host "Ingest error: $_"
    Write-Host "You can ingest later from the web UI button or: Invoke-RestMethod -Uri http://127.0.0.1:$port/ingest/doc -Method Post -Body '{}' -ContentType application/json"
}

Write-Host ""
Write-Host "Open in browser: http://127.0.0.1:$port"
Write-Host "Press Ctrl+C to stop — this will terminate the background uvicorn process."
try {
    Wait-Process -Id $proc.Id
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}
