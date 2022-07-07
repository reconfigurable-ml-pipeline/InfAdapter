using tensorflow serving multi-version serving feature, with model accepting batch and base64 image as input
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


### Steps to make the used Docker image
1. Install tensorflow
```shell
pip install tensorflow==2.9.1
pip install tensorflow_probability==0.17.0
```

2. Install pytorch
```shell
pip install torch==1.10.1
pip install torchvision==0.11.2
```

3. Convert Pytorch pretrained resnet models to Tensorflow saved model
```
python torch_to_tf.py
```


4. Add preprocessing to the generated Tensorflow models to allow them accept base64 image as input
```shell
mkdir tensorflow_b64
python tensorflow_b64.py
```

5. Run Tensorflow serving container
```shell
docker run -d --name serving_base tensorflow/serving:2.8.0
```

6. Copy model files to container file system
```shell
docker cp tensorflow_b64/ serving_base:/models/resnet
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
