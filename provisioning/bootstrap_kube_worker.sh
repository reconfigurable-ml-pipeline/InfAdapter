#!/bin/bash

apt install -qq -y sshpass >/dev/null 2>&1
sshpass -p "admin" scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no kube_master:/join_to_kube_cluster.sh /join_to_kube_cluster.sh 2>/dev/null
bash /join_to_kube_cluster.sh >/dev/null 2>&1
echo "Worker joined to cluster"
