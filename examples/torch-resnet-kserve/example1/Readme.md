Kserve Torchvision Resnet Models Pipeline (transformer + predictor)
-

1. build transformer image
```shell
docker build -f transformer/Dockerfile -t kserve-torch-resnet-transformer:v1 transformer
```
2. build predictor image
```shell
docker build -f predictor/Dockerfile -t kserve-torch-resnet-predictor:v1 predictor
```
3. deploy inference service to kubernetes
```shell
kubectl apply -f inference_service.yaml
```
4. edit service type to NodePort
```shell
kubectl edit svc torch-resnet-pipeline-transformer-default
```
    find type and change from ClusterIP to NodePort
5. get node_port
```shell
NODE_PORT=$(kubectl get svc torch-resnet-transformer-default -o jsonpath="{.spec.ports[0].nodePort}")
```
6.
```shell
WORKER_IP=$(kubectl get node --selector='!node-role.kubernetes.io/master' -o jsonpath="{.items[0].status.addresses[0].address}")
```
7. after the service is in running state, test it
```shell
curl ${WORKER_IP:1:-1}:$NODE_PORT/v1/models/kserve_resnet:predict -d @./input.json
```
