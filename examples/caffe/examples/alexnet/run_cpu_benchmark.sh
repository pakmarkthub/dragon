#!/bin/bash

for batchsize in 4096 2048 1024 512 256
do
    echo "===> Running AlexNet with batch size: $batchsize"
    sed "s/{{ BATCH_SIZE }}/$batchsize/g" examples/alexnet/fread_benchmark_train.prototxt.template > examples/alexnet/fread_benchmark_train.prototxt
    ../../experiments/drop-caches.sh
    time ./build/tools/caffe train --solver=examples/alexnet/cpu_benchmark_solver.prototxt -enable_mmap=true -dragon_tmp_folder=/opt/fio/pak/tmp/
done
