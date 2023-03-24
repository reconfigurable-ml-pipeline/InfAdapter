*InfAdapter: An Adaptation Mechanism for ML Inference Services*
-

<img src="architecture.png" alt="InfAdapter Structure" style="width:600px;"/>
<img src="adapter-component.png" alt="InfAdapter Structure" style="width:600px;"/>

## Instructions
1. Create a Kubernetes cluster
    1. Create a K8s cluster using Microk8s: [Get started](https://microk8s.io/docs/getting-started)
    2. Add another node to the k8s cluster: [Create a MicroK8s cluster](https://microk8s.io/docs/clustering)

2. Set up Prometheus monitoring inside the cluster (Todo: how?)

3. Create a namespace called mehran: `kubectl create ns mehran`

4. Build resnet models for TensorFlow Serving: instructions at [here](./examples/torch-resnet-kserve/example6/build_models.md)

5. Configure NFS server to keep and serve our models:
insructions at [here](./examples/torch-resnet-kserve/example6/build_nfs_server.md)

6. Export a cluster node's IP: `export CLUSTER_NODE_IP=NODE_IP`

7. Export NFS server IP: `export NFS_SERVER=NFS_SERVER_IP` (If not set, the same above CLUSTER_NODE_IP will be used)

8. Cache Docker images (Run and wait for "OK" message): `python auto_tuner/cache_images.py`

    ...


## Technology Stack
- Python
- Kubernetes
- TensorFlow Serving
- Prometheus
