#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage:"
    echo "./start.sh /path/to/data"
    exit -1
fi

docker run --gpus all --rm -it -d \
    --name hero \
    -v $(realpath $(dirname $0))/../:/home/docker_hero \
    -v $(realpath $1):/data \
    --shm-size 16G \
    --ipc=host -p 6006:80 \
    hero:latest
