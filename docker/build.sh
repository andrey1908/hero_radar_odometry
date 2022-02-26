#!/bin/bash

cd $(dirname $0)

NUM_THREADS=${1:-1}

docker build . \
    -f Dockerfile \
    --build-arg UID=$(id -g) \
    --build-arg GID=$(id -g) \
    --build-arg NUM_THREADS=${NUM_THREADS} \
    -t hero
