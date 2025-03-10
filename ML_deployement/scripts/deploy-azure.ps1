# Variables
$SUBSCRIPTION = "a5b55387-1733-4297-aaea-9dafbfcab6ed"
$RESOURCE_GROUP = "powerop-rg"
$ACR_RESOURCE_GROUP = "HVAC"
$AKS_CLUSTER = "powerop-aks"
$ACR_NAME = "hvac040325"

# Login and set subscription
Write-Host "Setting up Azure authentication..."
$account = az account show 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Login required..."
    az login --use-device-code
}
az account set --subscription $SUBSCRIPTION

# Get existing resource group location
Write-Host "Getting resource group details..."
$LOCATION = $(az group show --name $RESOURCE_GROUP --query location -o tsv)
Write-Host "Using location: $LOCATION"

# Get existing ACR details
Write-Host "Getting ACR details..."
$ACR_ID = $(az acr show --name $ACR_NAME --resource-group $ACR_RESOURCE_GROUP --query id -o tsv)
$ACR_LOGIN_SERVER = $(az acr show --name $ACR_NAME --resource-group $ACR_RESOURCE_GROUP --query loginServer -o tsv)

if (-not $ACR_ID) {
    Write-Error "Failed to get ACR details. Please check ACR name and resource group."
    exit 1
}

# Create AKS cluster with identity
Write-Host "Creating AKS cluster..."
try {
    Write-Host "Creating managed identity for AKS..."
    $IDENTITY_NAME = "powerop-aks-identity"
    az identity create --name $IDENTITY_NAME --resource-group $RESOURCE_GROUP
    $IDENTITY_ID = $(az identity show --name $IDENTITY_NAME --resource-group $RESOURCE_GROUP --query id -o tsv)
    $IDENTITY_CLIENT_ID = $(az identity show --name $IDENTITY_NAME --resource-group $RESOURCE_GROUP --query clientId -o tsv)

    Write-Host "Creating AKS cluster with managed identity..."
    az aks create `
        --resource-group $RESOURCE_GROUP `
        --name $AKS_CLUSTER `
        --node-count 1 `
        --node-vm-size Standard_DS2_v2 `
        --location $LOCATION `
        --generate-ssh-keys `
        --enable-managed-identity `
        --assign-identity $IDENTITY_ID `
        --assign-kubelet-identity $IDENTITY_ID

    # Wait for cluster creation
    Write-Host "Waiting for cluster creation..."
    do {
        Start-Sleep -Seconds 30
        $status = $(az aks show --name $AKS_CLUSTER --resource-group $RESOURCE_GROUP --query provisioningState -o tsv)
        Write-Host "Cluster status: $status"
        if ($status -eq "Succeeded") { break }
    } while ($true)

    # Grant ACR pull access
    Write-Host "Granting ACR pull access..."
    az role assignment create `
        --assignee $IDENTITY_CLIENT_ID `
        --role AcrPull `
        --scope $ACR_ID

}
catch {
    Write-Error "Failed to create AKS cluster: $_"
    exit 1
}

# Get AKS credentials
Write-Host "Getting AKS credentials..."
az aks get-credentials `
    --resource-group $RESOURCE_GROUP `
    --name $AKS_CLUSTER `
    --overwrite-existing

# Verify cluster access
Write-Host "Verifying cluster access..."
$retry = 0
do {
    kubectl cluster-info
    if ($LASTEXITCODE -eq 0) { break }
    $retry++
    if ($retry -gt 5) {
        Write-Error "Failed to connect to cluster after 5 attempts"
        exit 1
    }
    Start-Sleep -Seconds 10
} while ($true)

# Create namespace and secrets
Write-Host "Creating namespace and secrets..."
kubectl create namespace powerop
kubectl create secret docker-registry acr-auth `
    --namespace powerop `
    --docker-server=$ACR_LOGIN_SERVER `
    --docker-username=$ACR_NAME `
    --docker-password=$(az acr credential show `
        --name $ACR_NAME `
        --resource-group $ACR_RESOURCE_GROUP `
        --query "passwords[0].value" -o tsv)

kubectl create secret generic powerop-secrets `
    --namespace powerop `
    --from-literal=GROQ_API_KEY="$env:GROQ_API_KEY" `
    --from-literal=ASTRA_DB_TOKEN="$env:ASTRA_DB_TOKEN" `
    --from-literal=ASTRA_DB_API_ENDPOINT="$env:ASTRA_DB_API_ENDPOINT"

# Push image to ACR
Write-Host "Pushing image to ACR..."
az acr login --name $ACR_NAME
docker tag jshagarwal/powerop-ml:1.0.0 "$ACR_LOGIN_SERVER/powerop-ml:1.0.0"
docker push "$ACR_LOGIN_SERVER/powerop-ml:1.0.0"

# Deploy application
Write-Host "Deploying application..."
$deployment = Get-Content ./k8s/azure-deployment.yaml -Raw
$deployment = $deployment -replace 'hvac040325.azurecr.io', $ACR_LOGIN_SERVER
$deployment = $deployment -replace 'imagePullSecrets:.*', "imagePullSecrets:`n      - name: acr-auth"
$deployment | Set-Content ./k8s/azure-deployment.temp.yaml
kubectl apply -f ./k8s/azure-deployment.temp.yaml --namespace powerop
Remove-Item ./k8s/azure-deployment.temp.yaml

# Print summary
Write-Host "`nDeployment Summary:"
Write-Host "ACR: $ACR_LOGIN_SERVER"
Write-Host "AKS Cluster: $AKS_CLUSTER"
Write-Host "Resource Group: $RESOURCE_GROUP"
Write-Host "Location: $LOCATION"
Write-Host "Namespace: powerop"

# Get service IP
Write-Host "`nWaiting for service IP..."
kubectl get service powerop-ml --namespace powerop --watch
