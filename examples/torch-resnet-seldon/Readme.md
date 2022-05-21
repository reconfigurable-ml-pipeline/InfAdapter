### An example model deployment using torch and seldon-core

### prerequisites:
- Docker
- a Kubernetes cluster >= 1.18
- helm >= 3.0
### steps:
-   ```shell
    kubectl create namespace seldon-system
    ```
-   ```shell
    helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --namespace seldon-system
    ```
- Deploy the service to Kubernetes:
    ```shell 
    kubectl apply -f deploy.yml
    ```
- expose the deployment. example: 
    ```shell
    kubectl expose deployment torch-resnet-torch-resnet-0-torch-resnet --type NodePort --port 9000 --target-port 9000 --name torch-resnet-svc
    ```
- get nodePort:
    ```shell
    SERVICE_NODE_PORT=$(kubectl get svc torch-resnet-svc -o jsonpath="{.spec.ports[0].nodePort}")
    ```
- get IP address of a worker node
  ```shell
  WORKER_IP=$(kubectl get node --selector='!node-role.kubernetes.io/master' -o jsonpath="{.items[0].status.addresses[0].address}")
  ```
- test the deployment:
    ```shell
    curl ${WORKER_IP:1:-1}:$SERVICE_NODE_PORT/health/status
    ```