# Azure Setup Script for PowerOP_ML

# Install required modules
Write-Host "Installing required modules..."
Install-Module -Name Az -Scope CurrentUser -Repository PSGallery -Force

# Login to Azure
Write-Host "Logging into Azure..."
Connect-AzAccount

# Set variables
$resourceGroup = "powerop-ml-rg"
$location = "eastus"
$mlWorkspaceName = "powerop-ml-workspace"
$webAppName = "powerop-ml-app"

# Create Resource Group
Write-Host "Creating Resource Group..."
New-AzResourceGroup -Name $resourceGroup -Location $location

# Create ML Workspace
Write-Host "Creating ML Workspace..."
New-AzMLWorkspace -Name $mlWorkspaceName -ResourceGroupName $resourceGroup -Location $location

# Create Web App
Write-Host "Creating Web App..."
New-AzWebApp -Name $webAppName -ResourceGroupName $resourceGroup -Location $location -Sku "F1"

Write-Host "Setup completed! Resources created:"
Write-Host "- Resource Group: $resourceGroup"
Write-Host "- ML Workspace: $mlWorkspaceName"
Write-Host "- Web App: $webAppName"
