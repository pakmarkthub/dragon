#!/bin/bash

rmmod nvidia-uvm
modprobe nvidia-uvm

for num in 18 32 50 101 152
do
    echo "===> Running ResNet-$num uvm"
    sed "s/{{ INPUT_LAYER_TYPE }}/DragonFreadData/g" resnet_${num}/resnet_${num}.prototxt.template > resnet_${num}/resnet_${num}.prototxt
    ../../../../experiments/drop-caches.sh
    time ../../build/tools/caffe train --solver=resnet_${num}/resnet_${num}_solver.prototxt -enable_uvm=true
done
