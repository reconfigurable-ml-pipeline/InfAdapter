using tensorflow serving multi version serving feature
-

### Steps to deploy and test
1. Install requirements
```shell
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

6. Switch to another model
```shell
curl $WORKER_IP:$NODE_PORT/v3/models/kserve_resnet:switch-model -d '{"model": "resnet18"}'
```

7. Repeat step 7 to test again


### Steps to make the used image
1. Install tensorflow
```shell
pip install tensorflow==2.9.1
```

2. Install onnx
```shell
pip install onnx==1.11.0
pip install onnx_tf==1.10.0
```

3. Download resnet onnx files
```shell
mkdir onnx
cd onnx
for v in 18 34 50 101 152; do wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet$v-v1-7.onnx; done
```

4. Convert to tensorflow format
```shell
cd ..
mkdir tensorflow
python onnx_to_tf.py
```

5. Run tensorflow serving container
```shell
docker run -d --name serving_base tensorflow/serving
```

6 Copy model files to container file system
```shell
ocker cp tensorflow/ serving_base:/models
```

7. Commit the changes to create the Docker image
```shell
docker commit --change "ENV MODEL_NAME resnet" serving_base $USER/resnet_serving
```

8. Stop the serving_base container
```shell
docker kill serving_base
docker rm serving_base
```
