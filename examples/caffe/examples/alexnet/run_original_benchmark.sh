#!/bin/bash

for batchsize in 256 512 1024 2048 4096
do
    echo "===> Running AlexNet with batch size: $batchsize"
    sed "s/{{ BATCH_SIZE }}/$batchsize/g" examples/alexnet/original_train_val.prototxt.template > examples/alexnet/original_train_val.prototxt
    ../../experiments/drop-caches.sh
    echo "===> Start time: $(date +%s)"
    ./build/tools/caffe train --solver=examples/alexnet/original_solver.prototxt 
    echo "===> End time: $(date +%s)"
done
