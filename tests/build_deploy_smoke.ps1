Write-Host "Building Docker image..."

docker build -t cats-dogs-api:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed"
    exit 1
}

Write-Host "Starting container..."

docker rm -f cats-dogs-api 2>$null

docker run -d `
    --name cats-dogs-api `
    -p 8000:8000 `
    cats-dogs-api:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Container failed to start"
    exit 1
}

Write-Host "Waiting for application..."
Start-Sleep -Seconds 5

Write-Host "Running smoke tests..."

Set-Location tests
python test_post_deploy_smoke.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Smoke tests FAILED"

    docker logs cats-dogs-api

    docker rm -f cats-dogs-api

    exit 1
}

Write-Host "✅ Deployment and smoke tests successful"

Set-Location ..
docker rm -f cats-dogs-api
exit 0