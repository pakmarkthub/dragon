#!/bin/bash

rmmod nvidia-uvm

for num in 18 32 50 101 152
do
    echo "===> Running ResNet-$num cpu"
    sed "s/{{ INPUT_LAYER_TYPE }}/DragonMmapData/g" resnet_${num}/resnet_${num}.prototxt.template > resnet_${num}/resnet_${num}.prototxt
    ../../../../experiments/drop-caches.sh
    time ../../build/tools/caffe train --solver=resnet_${num}/resnet_${num}_solver.prototxt -enable_mmap=true -dragon_tmp_folder=/mnt/nvme/pak/tmp/
done
