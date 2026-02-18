#!/bin/bash

# Script de deploy no Kubernetes com Helm
set -e

echo "🚀 Iniciando deploy do NEXUS no Kubernetes..."

# Namespace
NAMESPACE="nexus-ai"
kubectl create namespace $NAMESPACE 2>/dev/null || true

# ConfigMaps e Secrets
echo "📦 Configurando secrets..."
kubectl create secret generic nexus-secrets \
  --namespace=$NAMESPACE \
  --from-env-file=.env.production \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy com Helm
echo "📦 Instalando com Helm..."
helm upgrade --install nexus ./helm/nexus \
  --namespace $NAMESPACE \
  --values ./helm/nexus/values.production.yaml \
  --wait \
  --timeout 10m

# Verificar status
echo "✅ Verificando status dos pods..."
kubectl get pods -n $NAMESPACE -w

# Configurar HPA (Horizontal Pod Autoscaling)
echo "📊 Configurando autoscaling..."
kubectl autoscale deployment nexus-api \
  --namespace=$NAMESPACE \
  --cpu-percent=70 \
  --min=3 \
  --max=10

# Service Mesh (Istio)
echo "🔀 Configurando service mesh..."
kubectl apply -f ./k8s/istio/virtual-service.yaml
kubectl apply -f ./k8s/istio/destination-rule.yaml

# Monitoring
echo "📈 Configurando monitoring..."
kubectl apply -f ./k8s/monitoring/service-monitor.yaml
kubectl apply -f ./k8s/monitoring/grafana-dashboard.yaml

# Backup automático
echo "💾 Configurando backup..."
kubectl apply -f ./k8s/backup/cronjob.yaml

echo "✅ Deploy concluído com sucesso!"
echo "📝 Acesse: https://nexus.yourdomain.com"