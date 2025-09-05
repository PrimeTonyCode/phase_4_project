# PowerShell test script for API
# This script doesn't need execution policy changes to run

$Url = "http://127.0.0.1:8000"

Write-Host "Testing BERT model API..."

# Test health endpoint
Write-Host "`nTesting health endpoint..."
try {
    $response = Invoke-WebRequest -Uri "$Url/health" -Method Get
    Write-Host "Response: $($response.Content)"
} catch {
    Write-Host "Error accessing health endpoint: $_"
}

# Test predict endpoint with sample tweets
Write-Host "`nTesting predict endpoint..."
$body = @{
    tweets = @(
        "I love this product! It works amazingly well.",
        "This is the worst purchase I have ever made."
    )
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "$Url/predict" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Response: $($response.Content)"
} catch {
    Write-Host "Error accessing predict endpoint: $_"
}

Write-Host "`nTest completed!"
