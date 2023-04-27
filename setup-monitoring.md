## Prometheus and Grafana for microk8s cluster

1. Just use the builtin add-on
```
microk8s enable prometheus
```
2. For enabling outside access make the `service/prometheus-k8s` and `service/grafana` of type NodePort isntead of ClusterIP using the following command and editing the `type` field. Also set the `web` nodePort for prometheus to 30090 and `http` port for grafana to 30300
```
kubectl edit svc prometheus-k8s -n monitoring
kubectl edit svc grafana -n monitoring
```
3. Find the Prometheus and grafana node ports using

For prometheus
```
kubectl get service prometheus-k8s -n monitoring -o jsonpath="{.spec.ports[0].nodePort}"
```
The output should be:
```
30090% 
```

For Grafana
```
kubectl get service grafana -n monitoring -o jsonpath="{.spec.ports[0].nodePort}"
```
The output should be:
```
30300% 
```


default credentials for accesing Grafana are:
```
username: admin
password: admin
```

4. Both of the grafana and prometheus are now accessible via the following links
```
<your node ip>:<prometheus port>
<your node ip>:<grafana port>
```
  5. To monitor Tensorflow serving, manually add a label "model_server: tfserving" to your Kubernetes Pods/Deployments, and apply the  PodMonitor bellow:
 ```
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: tfserving-podmonitor
  namespace: monitoring
  labels:
    podmonitor: tfserving
spec:
  namespaceSelector:
    any: true
  podMetricsEndpoints:
  - interval: 1s
    path: /monitoring/prometheus/metrics
  selector:
    matchLabels:
      model_server: tfserving
EOF
```
