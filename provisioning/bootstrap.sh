#!/bin/bash

# Delete swap entry from /etc/fstab
sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
# Turn swap off
swapoff -a
echo "Disabled swap"

systemctl disable --now ufw >/dev/null 2>&1
echo "Disabled ufw firewall"

cat >>/etc/modules-load.d/containerd.conf<<EOF
overlay
br_netfilter
EOF
modprobe overlay
modprobe br_netfilter
echo "Enabled overlay and br_netfilter kernel modules"

cat >>/etc/sysctl.d/kubernetes.conf<<EOF
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables  = 1
net.ipv4.ip_forward                 = 1
EOF
sysctl --system >/dev/null 2>&1
echo "Configured sysctl for iptables bridge"

apt update -qq >/dev/null 2>&1
apt install -qq -y containerd wget curl apt-transport-https >/dev/null 2>&1
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
systemctl enable containerd >/dev/null 2>&1
systemctl restart containerd
echo "Installed containerd"

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - >/dev/null 2>&1
apt-add-repository "deb http://apt.kubernetes.io/ kubernetes-xenial main" >/dev/null 2>&1
echo "Added kubernetes repository to apt"

apt install -qq -y kubeadm=1.23.5-00 kubelet=1.23.5-00 kubectl=1.23.5-00 >/dev/null 2>&1
echo "Installed kubeadm, kubelet and kubectl version 1.23.5"

sed -i 's/^PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config
echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
systemctl reload sshd
echo "Enabled logging in using password"

echo -e "admin\nadmin" | passwd root >/dev/null 2>&1
echo "export TERM=xterm" >> /etc/bash.bashrc
echo "Set root password"

cat >>/etc/hosts<<EOF
172.130.1.100   kube_master
172.130.1.101   kube_worker1
172.130.1.102   kube_worker2
EOF
echo "Added all cluster nodes to /etc/hosts"
