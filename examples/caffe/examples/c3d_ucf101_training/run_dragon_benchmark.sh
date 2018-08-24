#!/bin/bash

cd ../../../../kernel/ && ./reinsert-mod.sh
cd -

#for length in 12 24 36 48 60 72 84
for length in 72 84 48
do
    echo "===> Converting data ${length}"
    sed "s/{{ LENGTH }}/$length/g" convert_data.prototxt.template > convert_data.prototxt
    ../../build/tools/caffe train --solver=convert_data_solver.prototxt

    for enable_dragon in "true"
    do
        for readahead_type in "norm"
        do
            echo "===> Running C3D nvmgpu ${length} ${enable_dragon} ${readahead_type}"

            sed "s/{{ INPUT_LAYER_TYPE }}/DragonData/g" train_simple.prototxt.template > train_simple.prototxt.tmp
            sed "s/{{ LENGTH }}/$length/g" train_simple.prototxt.tmp > train_simple.prototxt
            rm train_simple.prototxt.tmp
            dmesg -C
            ../../../../experiments/drop-caches.sh
            time DRAGON_READAHEAD_TYPE=${readahead_type} DRAGON_NR_RESERVED_PAGES=524288 ../../build/tools/caffe train --solver=solver.prototxt -enable_dragon=${enable_dragon} -dragon_tmp_folder=/mnt/nvme/pak/tmp/
            #time DRAGON_READAHEAD_TYPE=${readahead_type} ../../build/tools/caffe train --solver=solver.prototxt -enable_uvm=${enable_dragon} -dragon_tmp_folder=/mnt/nvme/pak/tmp/
            dmesg
        done
    done
done
