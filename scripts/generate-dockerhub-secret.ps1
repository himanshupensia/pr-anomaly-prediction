# Docker Hub Registry Secret Generator for SAP AI Launchpad
# ==========================================================
# Generates the JSON payload required for Docker Registry Secret in AI Launchpad
#
# Usage:
#   .\ai\scripts\generate-dockerhub-secret.ps1 -DockerHubUser himanshupensia -AccessToken dckr_pat_xxx
#
# Prerequisites:
# 1. Docker Hub account
# 2. Docker Hub access token (create at https://hub.docker.com/settings/security)
#
# Output format for AI Launchpad:
# {
#   "name": "docker-him",
#   "data": {
#     ".dockerconfigjson": "<base64-encoded-docker-config>"
#   }
# }

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerHubUser,
    [Parameter(Mandatory=$true)]
    [string]$AccessToken,
    [string]$SecretName = "docker-him",
    [string]$OutputFile = ""
)

$ErrorActionPreference = "Stop"

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Docker Hub Registry Secret Generator" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Docker Hub User: $DockerHubUser" -ForegroundColor Gray
Write-Host "  Secret Name:     $SecretName" -ForegroundColor Gray
Write-Host ""

# Create Docker config JSON structure
# Format: {"auths":{"https://index.docker.io/v1/":{"username":"...","password":"...","auth":"..."}}}
$authString = "${DockerHubUser}:${AccessToken}"
$authBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($authString))

$dockerConfig = @{
    auths = @{
        "https://index.docker.io/v1/" = @{
            username = $DockerHubUser
            password = $AccessToken
            auth     = $authBase64
        }
    }
}

# Convert to JSON (compact, no extra whitespace)
$dockerConfigJson = $dockerConfig | ConvertTo-Json -Depth 5 -Compress

# Base64 encode the entire docker config JSON
$dockerConfigBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($dockerConfigJson))

# Create the AI Launchpad secret payload
$secretPayload = @{
    name = $SecretName
    data = @{
        ".dockerconfigjson" = $dockerConfigBase64
    }
}

# Convert to JSON with proper formatting
$secretJson = $secretPayload | ConvertTo-Json -Depth 3

Write-Host "Generated Secret Payload:" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host $secretJson
Write-Host "============================================`n" -ForegroundColor Cyan

# Save to file
if (-not $OutputFile) {
    $OutputFile = "docker-him-secret.json"
}

$secretJson | Out-File -FilePath $OutputFile -Encoding UTF8 -NoNewline

Write-Host "Secret saved to: $OutputFile" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps for AI Launchpad:" -ForegroundColor Yellow
Write-Host "  1. Go to AI Launchpad > SAP AI Core Administration" -ForegroundColor White
Write-Host "  2. Select your AI Core connection" -ForegroundColor White
Write-Host "  3. Go to Docker Registry Secrets" -ForegroundColor White
Write-Host "  4. Click 'Add' button" -ForegroundColor White
Write-Host "  5. Paste the JSON content above (or upload the file)" -ForegroundColor White
Write-Host "  6. Click 'Add'" -ForegroundColor White
Write-Host ""
Write-Host "Important:" -ForegroundColor Red
Write-Host "  - Secret name must match: $SecretName" -ForegroundColor Yellow
Write-Host "  - This matches imagePullSecrets in your workflow YAMLs" -ForegroundColor Yellow
Write-Host "  - Do NOT commit the generated JSON file to Git!" -ForegroundColor Yellow
Write-Host ""