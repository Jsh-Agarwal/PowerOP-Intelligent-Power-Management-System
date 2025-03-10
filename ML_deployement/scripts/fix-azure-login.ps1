# Clear Azure CLI credentials and cache
Write-Host "Clearing Azure CLI credentials..."
az logout

Write-Host "Removing Azure CLI cache..."
Remove-Item -Path "$env:USERPROFILE\.azure\msal_token_cache.bin" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$env:USERPROFILE\.azure\azureProfile.json" -Force -ErrorAction SilentlyContinue

Write-Host "Please login with your student email account..."
Write-Host "Opening browser for authentication..."
az login --allow-no-subscriptions

Write-Host "Checking for subscriptions..."
az account list --output table

Write-Host "If no subscriptions appear, please visit Azure for Students portal to complete registration"
Write-Host "https://azure.microsoft.com/free/students"
