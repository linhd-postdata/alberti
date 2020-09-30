#!/bin/bash
sudo mkdir -p /shared
sudo mount ${NFS-10.139.154.226:/shared} /shared
sudo chmod go+rw /shared
df -h --type=nfs
