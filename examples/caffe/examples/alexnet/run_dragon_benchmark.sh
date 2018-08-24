#!/bin/bash

for batchsize in 256 512 # 1024 2048 4096
do
    echo "===> Running AlexNet with batch size: $batchsize"
    sed "s/{{ BATCH_SIZE }}/$batchsize/g" examples/alexnet/dragon_benchmark_train.prototxt.template > examples/alexnet/dragon_benchmark_train.prototxt
    dmesg -C
    ../../experiments/drop-caches.sh
    echo "===> Start time: $(date +%s)"
    #./build/tools/caffe train --solver=examples/alexnet/dragon_benchmark_solver.prototxt -enable_dragon=true -dragon_tmp_folder=/opt/fio/pak/tmp/
    ./build/tools/caffe train --solver=examples/alexnet/dragon_benchmark_solver.prototxt
    echo "===> End time: $(date +%s)"
    dmesg
done
