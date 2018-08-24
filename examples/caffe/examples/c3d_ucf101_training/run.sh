#!/bin/bash

rmmod nvidia-uvm

echo "===> Running original C3D"
./run_original_benchmark.sh

echo "===> Running uvm C3D"
./run_uvm_benchmark.sh

