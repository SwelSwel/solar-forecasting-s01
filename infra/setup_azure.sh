#!/usr/bin/env bash
# ==========================================================================
# Azure infrastructure setup for S01 Solar Forecast
# Run in Azure Cloud Shell (Bash) — takes ~10 minutes
#
# Creates: Resource Group, Container Registry, Container Apps environment,
#          Managed Identity, App Registration with OIDC federated credentials
# ==========================================================================

set -euo pipefail

# --- Edit these for your project ---
LOCATION="eastasia"
RG="rg-solar-forecast"
ACR="acrsolarforecast01"
ACA_ENV="aca-solar-env"
LOG="law-solar"
PULL_ID="id-solar-acrpull"
APP="solar-forecast"
GITHUB_REPO="SwelSwel/solar-forecasting-s01"
# ------------------------------------

echo "============================================"
echo " S01 Solar Forecast — Azure Setup"
echo "============================================"
echo "Location:    $LOCATION"
echo "RG:          $RG"
echo "ACR:         $ACR"
echo "App:         $APP"
echo "GitHub:      $GITHUB_REPO"
echo ""

# ------------------------------------------------------------------
# Step 1: Resource Group
# ------------------------------------------------------------------
echo ">>> Step 1: Resource Group"
az group create --name "$RG" --location "$LOCATION" -o none
echo "    Created: $RG"

# ------------------------------------------------------------------
# Step 2: Container Registry
# ------------------------------------------------------------------
echo ">>> Step 2: Container Registry"
az acr create --resource-group "$RG" --name "$ACR" --sku Basic --admin-enabled false -o none
ACR_ID=$(az acr show --name "$ACR" --query id -o tsv)
echo "    Created: $ACR ($ACR_ID)"

# ------------------------------------------------------------------
# Step 3: Container Apps Environment + Log Analytics
# ------------------------------------------------------------------
echo ">>> Step 3: Container Apps Environment (this may take several minutes)"
az monitor log-analytics workspace create \
  --resource-group "$RG" --workspace-name "$LOG" -o none

LOG_ID=$(az monitor log-analytics workspace show \
  --resource-group "$RG" --workspace-name "$LOG" \
  --query customerId -o tsv)
LOG_KEY=$(az monitor log-analytics workspace get-shared-keys \
  --resource-group "$RG" --workspace-name "$LOG" \
  --query primarySharedKey -o tsv)

az containerapp env create \
  --name "$ACA_ENV" --resource-group "$RG" --location "$LOCATION" \
  --logs-workspace-id "$LOG_ID" --logs-workspace-key "$LOG_KEY" -o none
echo "    Created: $ACA_ENV"

# ------------------------------------------------------------------
# Step 4: Managed Identity for ACR Pull
# ------------------------------------------------------------------
echo ">>> Step 4: Managed Identity"
az identity create --name "$PULL_ID" --resource-group "$RG" -o none
PULL_IDENTITY_ID=$(az identity show --name "$PULL_ID" --resource-group "$RG" --query id -o tsv)
PULL_PRINCIPAL=$(az identity show --name "$PULL_ID" --resource-group "$RG" --query principalId -o tsv)

# Wait for identity propagation
echo "    Waiting 30s for identity propagation..."
sleep 30

az role assignment create \
  --assignee "$PULL_PRINCIPAL" --role AcrPull --scope "$ACR_ID" -o none
echo "    Created: $PULL_ID with AcrPull on $ACR"

# ------------------------------------------------------------------
# Step 5: App Registration + OIDC Federated Credentials
# ------------------------------------------------------------------
echo ">>> Step 5: App Registration + OIDC"
APP_REG_NAME="gh-solar-forecast-deploy"

# Create app registration
APP_ID=$(az ad app create --display-name "$APP_REG_NAME" --query appId -o tsv)
echo "    App ID: $APP_ID"

# Create service principal
az ad sp create --id "$APP_ID" -o none 2>/dev/null || true
SP_OBJ=$(az ad sp show --id "$APP_ID" --query id -o tsv)

# Assign Contributor on the resource group
RG_ID=$(az group show --name "$RG" --query id -o tsv)
az role assignment create \
  --assignee "$SP_OBJ" --role Contributor --scope "$RG_ID" -o none

# Assign AcrPush so GitHub Actions can push images
az role assignment create \
  --assignee "$SP_OBJ" --role AcrPush --scope "$ACR_ID" -o none

# Federated credentials for GitHub Actions OIDC
TENANT_ID=$(az account show --query tenantId -o tsv)
SUB_ID=$(az account show --query id -o tsv)

# Main branch
az ad app federated-credential create --id "$APP_ID" --parameters "{
  \"name\": \"github-main\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:${GITHUB_REPO}:ref:refs/heads/main\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}" -o none

# Production environment
az ad app federated-credential create --id "$APP_ID" --parameters "{
  \"name\": \"github-production\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:${GITHUB_REPO}:environment:production\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}" -o none

echo "    Federated credentials created for main branch and production environment"

# ------------------------------------------------------------------
# Summary — save this output!
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo " SETUP COMPLETE — SAVE THESE VALUES"
echo "============================================"
echo ""
echo "GitHub Secrets (Settings → Secrets → Actions):"
echo "  AZURE_CLIENT_ID       = $APP_ID"
echo "  AZURE_TENANT_ID       = $TENANT_ID"
echo "  AZURE_SUBSCRIPTION_ID = $SUB_ID"
echo ""
echo "GitHub Variables (Settings → Variables → Actions):"
echo "  RESOURCE_GROUP    = $RG"
echo "  ACR_NAME          = $ACR"
echo "  ACA_ENV           = $ACA_ENV"
echo "  APP_NAME          = $APP"
echo "  PULL_IDENTITY_ID  = $PULL_IDENTITY_ID"
echo ""
echo "ACR Login Server: ${ACR}.azurecr.io"
echo ""
echo "Next steps:"
echo "  1. Create 'production' environment in GitHub repo settings"
echo "  2. Add the secrets and variables above"
echo "  3. Push to main — the workflow will build and deploy"
echo "============================================"
