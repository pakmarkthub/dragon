#!/bin/bash

#rsync -r --progress /opt/nvme/pak/ilsvrc12_train_lmdb_full /opt/fio/pak/ilsvrc12/ilsvrc12_train_lmdb_full/
#rsync -r --progress /opt/nvme/pak/ilsvrc12_val_lmdb_full /opt/fio/pak/ilsvrc12/ilsvrc12_val_lmdb_full/

cd ../../../../kernel/ && ./reinsert-mod.sh
cd -

for enable_dragon in "false" # "true"  
do
    for readahead_type in "norm" #"aggr"
    do
        for num in 18 #32 50 101 152
        do
            echo "===> Running ResNet-${num} nvmgpu ${enable_dragon} ${readahead_type}"
            sed "s/{{ INPUT_LAYER_TYPE }}/DragonData/g" resnet_${num}/resnet_${num}.prototxt.template > resnet_${num}/resnet_${num}.prototxt
            dmesg -C
            ../../../../experiments/drop-caches.sh
            #time DRAGON_READAHEAD_TYPE=aggr DRAGON_NR_RESERVED_PAGES=4194304 ../../build/tools/caffe train --solver=resnet_${num}/resnet_${num}_solver.prototxt -enable_dragon=true -dragon_tmp_folder=/mnt/nvme/pak/tmp/
            time DRAGON_READAHEAD_TYPE=${readahead_type} ../../build/tools/caffe train --solver=resnet_${num}/resnet_${num}_solver.prototxt -enable_dragon=${enable_dragon} -dragon_tmp_folder=/mnt/nvme/pak/tmp/
            #time ../../build/tools/caffe train --solver=resnet_${num}/resnet_${num}_solver.prototxt 
            dmesg
        done
    done
done
