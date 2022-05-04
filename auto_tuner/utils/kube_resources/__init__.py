from kubernetes import client, config

config.load_kube_config()
core_api = client.CoreV1Api()
apps_api = client.AppsV1Api()
autoscaling_api = client.AutoscalingV1Api()
custom_api = client.CustomObjectsApi()
