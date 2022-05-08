#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage:"
    echo "./start.sh /path/to/data"
    exit -1
fi

docker run --rm -it -d \
    --name hero \
    -v $(realpath $(dirname $0))/../:/home/docker_hero/hero_radar_odometry \
    -v $(realpath $1):/data \
    --shm-size 16G \
    --net "host" \
    --privileged \
    --gpus all \
    hero:latest
