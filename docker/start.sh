#!/bin/bash

docker run --gpus all --rm -it -d \
    --name hero \
    -v $(realpath $(dirname $0))/../:/home/docker_hero \
    -v /media/cds-jetson-host/data/oxford:/workspace \
    --shm-size 16G \
    --ipc=host -p 6006:80 \
    hero:latest
