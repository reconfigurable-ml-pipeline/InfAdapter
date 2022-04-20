#!/bin/bash

kubeadm config images pull >/dev/null 2>&1
echo "Pulled required images for kubernetes"

kubeadm init --kubernetes-version=v1.23.5 --apiserver-advertise-address=172.130.1.100 --pod-network-cidr=192.168.0.0/16 >> /var/log/kubeinit.log 2>/dev/null
echo "Initialized kubernetes cluster"

mkdir ~/.kube
touch ~/.kube/config
cp /etc/kubernetes/admin.conf ~/.kube/config
kubectl apply -f https://docs.projectcalico.org/v3.22/manifests/calico.yaml >/dev/null 2>&1
echo "Deployed a Calico network in cluster"

kubeadm token create --print-join-command > /join_to_kube_cluster.sh 2>/dev/null
echo "Created join command and saved in /join_to_kube_cluster.sh in order for workers to use"
