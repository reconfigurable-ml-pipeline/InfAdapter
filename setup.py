import os
from setuptools import setup, find_packages


def read():
    return open(os.path.join(os.path.dirname(__file__), "README.md")).read()


setup(
    name="auto_tuner",
    version="0.0.1",
    keywords=["ML Inference", "ML Service", "Autoscaling", "Auto-Configuration", "Kubernetes", "Cloud Computing"],
    packages=find_packages("."),
    long_description=read(),
    install_requires=[
        "gym==0.24.0",
        "ray[rllib]==1.9.0",
        "ray[serve]==1.9.0",
        "aioredis==1.3.1",
        "simpy==4.0.1",
        "matplotlib==3.5.2",
        "PyQt5==5.15.4",
        "tensorflow==2.7.0",
        "tensorflow-serving-api==2.7.0",
        "grpcio==1.39.0",
        "protobuf==3.20.0",
        "numpy==1.19.2",
        "pillow==8.3.2",
        "locust==2.9.0",
        "redis==3.5.3",
        "opencv-python==4.6.0.66",
        "kubernetes-python-client @ git+ssh://git@github.com/reconfigurable-ml-pipeline/kubernetes-python-client.git",
        "barazmoon @ git+ssh://git@github.com/reconfigurable-ml-pipeline/load_tester.git"
    ],

)
