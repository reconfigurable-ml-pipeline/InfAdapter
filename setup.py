import os
from setuptools import setup, find_packages


def read():
    return open(os.path.join(os.path.dirname(__file__), "README.md")).read()


setup(
    name="autoscaler",
    version="0.0.1",
    keyword="horizontal autoscaling kubernetes distributed systems reinforcement learning",
    packages=["autoscaler"],
    long_description=read(),
    install_requires=[
        "kubernetes==17.17.0",
        "gym==0.18.0",
        "ray[rllib]",
    ]
)
