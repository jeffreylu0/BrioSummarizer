#!/usr/bin/env bash

image="jeffreylu0/summarizer2"
user="ai-txg-1-user"

sudo yum update
sudo yum install docker

# Add group membership for user to run docker commands without sudo
sudo usermod -a -G docker $user && newgrp docker

sudo systemctl enable docker.service # enable docker service at boot time
sudo systemctl start docker.service # start docker service

docker pull $image
docker run -dp 80:80 $image
