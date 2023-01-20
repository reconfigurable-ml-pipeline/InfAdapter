1. `sudo apt update`
2. `sudo apt install nfs-kernel-server`
3. `sudo mkdir -p /fileshare`
4. `sudo chmod 777 /fileshare`
5. ```shell
    sudo -s
    echo "/fileshare *(rw,no_subtree_check,no_root_squash)" >> /etc/exports
    exit
    ```
6. `sudo exportfs -a`
7. `sudo systemctl restart nfs-kernel-server`
8. `sudo ufw allow 2049`