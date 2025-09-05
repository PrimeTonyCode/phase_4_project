# Test the BERT model API

param(
    [string]$Url = "http://localhost:8000"
)

Write-Host "Testing BERT model API..."

# Test health endpoint
Write-Host "`nTesting health endpoint..."
try {
    Invoke-RestMethod -Uri "$Url/health" -Method Get | ConvertTo-Json
} catch {
    Write-Host "Error accessing health endpoint: $_"
}

# Test predict endpoint
Write-Host "`nTesting predict endpoint..."
$body = @{
    tweets = @(
        "I love this product! It works amazingly well.",
        "This is the worst purchase I have ever made."
    )
} | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "$Url/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 4
} catch {
    Write-Host "Error accessing predict endpoint: $_"
}

Write-Host "`nTest completed!"
