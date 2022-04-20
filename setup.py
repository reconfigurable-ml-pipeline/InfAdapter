import os
from setuptools import setup


def read():
    return open(os.path.join(os.path.dirname(__file__), "README.md")).read()


setup(
    name="autoscaler",
    version="0.0.1",
    keyword="ML Inference Service Autoscaling Kubernetes Cloud Computing",
    packages=["autoscaler"],
    long_description=read(),
    install_requires=[
        "kubernetes==17.17.0",
        "gym==0.18.0",
        "ray[rllib]",
        "aioredis==1.3.1",
        "simpy==4.0.1",
        "matplotlib==3.4.2",
        "PyQt5==5.15.4",
        "tensorflow==2.6.0",
        "python-decouple==3.5",
    ]
)
