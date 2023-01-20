using tensorflow serving multi-version serving feature, with model accepting batch and base64 image as input, where models are read from nfs-server
-

### Steps to deploy and test
1. Install requirements
```shell
pip install opencv-python==4.6.0.66
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.7.1/cert-manager.yaml
wget https://github.com/kserve/kserve/releases/download/v0.8.0/kserve.yaml
```
> Edit kserve.yaml file, change Serverless to RawDeployment,
> then use kubectl apply -f kserve.yaml to deploy kserve on your cluster

2. Deploy configmap
```shell
kubectl apply -f models_config.yaml
```
3. Deploy inference service to kubernetes
```shell
kubectl apply -f inference_service.yaml
```
4. Create a NodePort service
```shell
kubectl expose deploy tfserving-resnet-predictor-default --target_port 8501 --type NodePort --name tfserving-resnet-svc
```

5. after the service is in running state, test it
```shell
python resnet_client.py
```

6. Switch to another model by editing the version field in configmap
```shell
kubectl edit cm tfserving-resnet-cm
```

7. Repeat step 7 to test again


### Steps build nfs server
> Follow the steps at [build_nfs_server](./build_nfs_server.md)


### Add models to nfs server to be used in Pods
> Follow the steps at [build_model](./build_models.md)