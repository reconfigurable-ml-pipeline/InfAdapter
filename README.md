*InfAdapter: An Adaptation Mechanism for ML Inference Services*
-

<img src="InfAdapter.png" alt="InfAdapter Structure" style="width:600px;"/>

## Instructions
X. Create a Kubernetes cluster

X. Set up Prometheus monitoring inside the cluster

X. Create a namespace called mehran: `kubectl create ns mehran`

X. Build resnet models for TensorFlow Serving: instructions at [here](./examples/torch-resnet-kserve/example6/build_models.md)

X. Configure NFS server to keep and serve our models:
insructions at [here](./examples/torch-resnet-kserve/example6/build_nfs_server.md)

X. Export NFS server IP: at your terminal, run:  `export NFS_SERVER=NFS_SERVER_IP` replacing the node's IP address with NFS_SERVER_IP

X. Export a cluster node's IP: `export CLUSTER_NODE_IP=NODE_IP`


    ...


## Technology Stack
- Python
- Kubernetes
- TensorFlow Serving
- Prometheus
