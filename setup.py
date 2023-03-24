import os
from setuptools import setup, find_packages


def read():
    return open(os.path.join(os.path.dirname(__file__), "README.md")).read()


setup(
    name="infadapter",
    version="0.0.1",
    keywords=["ML Inference", "ML Service", "Autoscaling", "Auto-Configuration", "Kubernetes", "Cloud Computing"],
    packages=find_packages("."),
    long_description=read(),
    install_requires=[
        "wheel==0.38.4",
        "matplotlib==3.6.3",
        "numpy==1.23.4",
        "tensorflow==2.11.0",
        "tensorflow-serving-api==2.11.0",
        "grpcio==1.43.0",
        "pandas==1.5.3",
        "scikit-learn==1.2.0",
        "kubernetes-python-client @ git+https://github.com/reconfigurable-ml-pipeline/kubernetes-python-client.git",
        "barazmoon @ git+https://github.com/reconfigurable-ml-pipeline/load_tester.git",
    ],
    extras_require={
        "dev": [
            "pillow==9.4.0",
            "opencv-python==4.6.0.66"
        ]
    }
)
