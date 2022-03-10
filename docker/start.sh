#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage:"
    echo "./start.sh /path/to/dataset"
    exit -1
fi

docker run --gpus all --rm -it -d \
    --name hero \
    -v $(realpath $(dirname $0))/../:/home/docker_hero \
    -v $(realpath $1):/dataset \
    --shm-size 16G \
    --net "host" \
    --privileged \
    hero:latest
