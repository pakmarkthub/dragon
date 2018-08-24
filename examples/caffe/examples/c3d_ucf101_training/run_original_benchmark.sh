#!/bin/bash

rmmod nvidia-uvm
modprobe nvidia-uvm

for length in 12 24 36 48 60 72 84
do
    echo "===> Converting data ${length}"
    sed "s/{{ LENGTH }}/$length/g" convert_data.prototxt.template > convert_data.prototxt
    ../../build/tools/caffe train --solver=convert_data_solver.prototxt

    echo "===> Running C3D cudamemcpy ${length}"
    sed "s/{{ INPUT_LAYER_TYPE }}/DragonFreadData/g" train_simple.prototxt.template > train_simple.prototxt.tmp
    sed "s/{{ LENGTH }}/$length/g" train_simple.prototxt.tmp > train_simple.prototxt
    rm train_simple.prototxt.tmp
    ../../../../experiments/drop-caches.sh
    time ../../build/tools/caffe train --solver=solver.prototxt
done
