<#
.SYNOPSIS
    Generates a Kubernetes imagePullSecret for Docker Hub (or any OCI registry)
    and registers it with SAP AI Core via the AI Core REST API.

.DESCRIPTION
    SAP AI Core needs credentials to pull your training and serving images from
    a private container registry.  This script:
        1. Base64-encodes your registry credentials.
        2. Creates the secret payload in Docker config JSON format.
        3. Registers the secret with SAP AI Core using the /v2/admin/dockerRegistrySecrets API.

    Prerequisites:
        - PowerShell 7+ (Windows / macOS / Linux)
        - An SAP BTP Service Key for AI Core (provides clientid, clientsecret, url, serviceurls)
        - A Docker Hub (or other OCI registry) account with push access

.PARAMETER AiCoreTokenUrl
    OAuth2 token endpoint from your AI Core service key.
    Example: https://your-subdomain.authentication.eu10.hana.ondemand.com/oauth/token

.PARAMETER AiCoreClientId
    clientid from the AI Core service key.

.PARAMETER AiCoreClientSecret
    clientsecret from the AI Core service key.

.PARAMETER AiCoreApiUrl
    Base URL of the AI Core API (serviceurls.AI_API_URL from the service key).
    Example: https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com

.PARAMETER ResourceGroup
    SAP AI Core resource group (default: "default").

.PARAMETER RegistryServer
    OCI registry server hostname (default: "docker.io").

.PARAMETER RegistryUsername
    Your Docker Hub username (or service-account username for other registries).

.PARAMETER RegistryPassword
    Your Docker Hub Personal Access Token or password.
    Best practice: pass via $env:REGISTRY_PASSWORD rather than the command line.

.PARAMETER SecretName
    Name of the secret as it will appear in SAP AI Core (default: "docker-registry-secret").

.EXAMPLE
    .\generate-dockerhub-secret.ps1 `
        -AiCoreTokenUrl    "https://mysubdomain.authentication.eu10.hana.ondemand.com/oauth/token" `
        -AiCoreClientId    "sb-abc123" `
        -AiCoreClientSecret $env:AI_CORE_CLIENT_SECRET `
        -AiCoreApiUrl      "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com" `
        -RegistryUsername  "mydockerhubuser" `
        -RegistryPassword  $env:REGISTRY_PASSWORD
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $AiCoreTokenUrl,
    [Parameter(Mandatory)][string] $AiCoreClientId,
    [Parameter(Mandatory)][string] $AiCoreClientSecret,
    [Parameter(Mandatory)][string] $AiCoreApiUrl,
    [string] $ResourceGroup    = "default",
    [string] $RegistryServer   = "docker.io",
    [Parameter(Mandatory)][string] $RegistryUsername,
    [Parameter(Mandatory)][string] $RegistryPassword,
    [string] $SecretName       = "docker-registry-secret"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Obtain AI Core OAuth2 token
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "`n[1/3] Fetching AI Core access token …" -ForegroundColor Cyan

$tokenBody = @{
    grant_type    = "client_credentials"
    client_id     = $AiCoreClientId
    client_secret = $AiCoreClientSecret
}

$tokenResponse = Invoke-RestMethod `
    -Uri    "$AiCoreTokenUrl?grant_type=client_credentials" `
    -Method Post `
    -Body   $tokenBody `
    -ContentType "application/x-www-form-urlencoded"

$accessToken = $tokenResponse.access_token
if (-not $accessToken) {
    throw "Failed to obtain access token from $AiCoreTokenUrl"
}
Write-Host "    Access token acquired." -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Build Docker config JSON payload
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "`n[2/3] Building Docker registry secret payload …" -ForegroundColor Cyan

$credentialRaw  = "${RegistryUsername}:${RegistryPassword}"
$credentialB64  = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($credentialRaw))

$dockerConfigObj = @{
    auths = @{
        $RegistryServer = @{
            username = $RegistryUsername
            password = $RegistryPassword
            auth     = $credentialB64
        }
    }
}
$dockerConfigJson = $dockerConfigObj | ConvertTo-Json -Depth 5 -Compress
$dockerConfigB64  = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($dockerConfigJson))

# SAP AI Core secret body
$secretPayload = @{
    name = $SecretName
    data = @{
        ".dockerconfigjson" = $dockerConfigB64
    }
} | ConvertTo-Json -Depth 5

Write-Host "    Payload ready for secret: $SecretName" -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Register secret with SAP AI Core
# ─────────────────────────────────────────────────────────────────────────────
Write-Host "`n[3/3] Registering secret with SAP AI Core …" -ForegroundColor Cyan

$secretsUrl = "$AiCoreApiUrl/v2/admin/dockerRegistrySecrets"
$headers = @{
    "Authorization"     = "Bearer $accessToken"
    "AI-Resource-Group" = $ResourceGroup
    "Content-Type"      = "application/json"
}

try {
    $response = Invoke-RestMethod `
        -Uri     $secretsUrl `
        -Method  Post `
        -Headers $headers `
        -Body    $secretPayload

    Write-Host "`n✔  Secret '$SecretName' registered successfully." -ForegroundColor Green
    Write-Host "   Use this name in your workflow YAML under 'imagePullSecrets':" -ForegroundColor Gray
    Write-Host "       - name: $SecretName" -ForegroundColor Yellow
    $response | ConvertTo-Json -Depth 5 | Write-Host
}
catch {
    $statusCode = $_.Exception.Response?.StatusCode?.value__
    if ($statusCode -eq 409) {
        Write-Host "`n⚠  Secret '$SecretName' already exists.  Updating …" -ForegroundColor Yellow
        Invoke-RestMethod `
            -Uri     "$secretsUrl/$SecretName" `
            -Method  Patch `
            -Headers $headers `
            -Body    $secretPayload | Out-Null
        Write-Host "✔  Secret updated." -ForegroundColor Green
    }
    else {
        Write-Error "Failed to register secret: $_"
    }
}
