#!/bin/bash

for batchsize in 256 512 1024 2048 4096
do
    echo "===> Running AlexNet with batch size: $batchsize"
    sed "s/{{ BATCH_SIZE }}/$batchsize/g" examples/alexnet/fread_benchmark_train.prototxt.template > examples/alexnet/fread_benchmark_train.prototxt
    ../../experiments/drop-caches.sh
    echo "===> Start time: $(date +%s)"
    ./build/tools/caffe train --solver=examples/alexnet/fread_benchmark_solver.prototxt 
    echo "===> End time: $(date +%s)"
done
