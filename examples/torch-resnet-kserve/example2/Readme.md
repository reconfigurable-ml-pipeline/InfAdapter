Kserve Torchvision Resnet Models With Model Switching
-

1. Install requirements
```shell
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.7.1/cert-manager.yaml
wget https://github.com/kserve/kserve/releases/download/v0.7.0/kserve.yaml
```
> Edit kserve.yaml file, change Serverless to RawDeployment and all v1alpha2 to v1

2. Deploy configmap
```shell
kubectl apply -f resnet_configmap.yaml
```
3. Deploy inference service to kubernetes
```shell
kubectl apply -f inference_service.yaml
```
4. Create a NodePort service
```shell
kubectl expose deploy torch-resnet-predictor-default --target_port 8080 --type NodePort --name torch-resnet-svc
```
    find type and change from ClusterIP to NodePort
5. Get node_port
```shell
NODE_PORT=$(kubectl get svc torch-resnet-svc -o jsonpath="{.spec.ports[0].nodePort}")
```
6. Get IP address of one of the worker nodes
```shell
WORKER_IP=$(kubectl get node --selector='!node-role.kubernetes.io/master' -o jsonpath="{.items[0].status.addresses[0].address}")
```
7. after the service is in running state, test it
```shell
curl $WORKER_IP:$NODE_PORT/v1/models/kserve_resnet:predict -d @./input.json
```

8. switch to another model
```shell
curl $WORKER_IP:$NODE_PORT/v3/models/kserve_resnet:switch-model -d '{"model": "resnet18"}'
```

9. repeat step 7 to test again