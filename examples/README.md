# Example Applications

This folder contains applications that have been integrated with DRAGON. We also
include the other versions we used to compare with the DRAGON-integrated
version. Please refer to our SC18 paper *DRAGON:
Breaking GPU Memory Capacity Limits with Direct NVM Access* for more details.

The directory structure of each application is as follows:

* **programs** contains the source code of that application.

    * **cudamemcpy** corresponds to the *Default* version. This version is
similar to the original version but the application reads and writes data
to/from files in the memory-dump format instead.

    * **hostreg** corresponds to the *Hostreg* version. This version uses
*cudaHostRegister()* along with *mmap()* to access data from files.

    * **uvm** corresponds to the *UM-P* version. This version uses POSIX-IO with
NVIDIA's Unified Memory.

    * **nvmgpu** corresponds to the *DRAGON* version.

* **scripts** contains essential scripts for reproducing our results we reported
in the paper.

## Getting Started

### Prerequisites

* Python2.7 or above
* matplotlib
* numpy
* scipy
* CUDA version 9.0 or above
* DRAGON library and driver
* [gpufs](https://github.com/gpufs/gpufs)

### Compiling

1. For *binomialOptions*, *BlackScholes*, and *vectorAdd*, go to the *scripts*
folder inside those application folder. Run the *prepare-programs* script. You
need to specify the CUDA samples location. This script supports only the CUDA
sample applications from CUDA version 9.0.

```
cd <application>/scripts
./prepare-programs /usr/local/cuda-9.0/samples
```

2. For other applications

```
cd <application>/programs
make -j
```

### How to run an application

This section provides steps for running an example application.

1. Go to the *scripts* folder of the application you want to run.

```
cd <application>/scripts
```

2. Generate data using the *gendata* script. An example command for generating
input data files on
*/tmp* folder is as shown below. When using the generated input files, the
maximum memory footprint of the application is 64 GiB. You can explore all
available options by calling *./gendata -h*.

```
./gendata /tmp item 64G
```

3. Run the application on the input data using the *run* script. An example to
run all available versions of the application using the above generated input
files is shown below. The output result is saved to the *output.log* file. You
can explore all available options by calling *./run -h*. Note that this command
requires root privilege in order to automatically switching between the DRAGON
driver and the original NVIDIA driver.

```
sudo su
../../../scripts/activate-dragon
./run --repeat 1 output.log
```

## How to reproduce the results in the paper (Evaluation Section)

This section gives you steps for running all example applications, collecting
and converting results, and generating the graphs.

1. Prepare your node. Make sure that your host memory capacity is slightly above
64 GiB and the swap space is disabled. We recommend you to set the host memory
capacity to around 80 GiB in order to leave room for other processes (daemon,
ssh, etc.). The following websites provide guideline on how to do them in
software.

* https://stackoverflow.com/questions/13484016/setting-limit-to-total-physical-memory-available-in-linux
* https://serverfault.com/questions/684771/best-way-to-disable-swap-in-linux

2. Make sure that your NVMe device is formatted with *ext4* and the device's
free space is more than 512 GiB.

3. Run all experiments. We provide a script to automatically do it for you. This
step may take several hours to a day. You may want to run execute it using
*screen*. Also, this script needs the *root* privilege to run.

```
sudo su
cd <dragon-root>/scripts
./activate-dragon
cd <dragon-root>/examples
./run <location-on-your-nvme>
```

4. Extract the results. You need to do this step for all applications

```
cd <dragon-root>/examples/<application>/results
python ../../analyzers/convert_result.py results.out cudamemcpy > result-cudamemcpy.data
python ../../analyzers/convert_result.py results.out hostreg > result-hostreg.data
python ../../analyzers/convert_result.py results.out uvm > result-uvm.data
python ../../analyzers/convert_result.py results.out nvmgpu > result-nvmgpu.data
```

* Additional steps for *hotspot* and *vectorAdd*

```
cd <dragon-root>/examples/<application>/results
python ../../analyzers/convert_result.py result-nvmgpu-rh-disable.out nvmgpu > result-nvmgpu-rh-disable.data
```

5. Generate graphs from the results.

```
cd <dragon-root>/examples/analyzers
python ptc.py   # Figure 3
python plot_compare_readahead.py    # Figure 5
```

## How to reproduce the results in the paper (Case Study Section)

This section provides steps for reproducing the results shown in the case study
section. The experiment mainly uses our customized *Caffe* that comes with this
repository: <dragon-root>/examples/caffe.

1. Compile Caffe. Please follow the official
[instruction](http://caffe.berkeleyvision.org/installation.html). Some important
points regarding this compilation step:

* Use the code provided in <dragon-root>/examples/caffe
* Build using *cmake* in a new folder <dragon-root>/examples/caffe/build
* Compile with *ATLAS*

2. Download [ILSVRC12](http://www.image-net.org/challenges/LSVRC/2012/) and
[UCF101](http://crcv.ucf.edu/data/UCF101.php) datasets.

3. Convert the datasets to *memory-dump* format. If your NVMe capacity is small,
you may want to convert only one dataset at a time and convert the other one
after you finish the corresponding experiment.

```
cd <dragon-root>/examples/caffe/scripts
./gendata c3d <path-to-ucf101> <folder-on-nvme-for-converted-ucf101-data>
./gendata resnet <path-to-ilsvrc12> <folder-on-nvme-for-converted-ilsvrc12-data>
```

4. Run the C3D and Resnet experiments. The automated script need root privilege.
One experiment may take several hours.

```
cd <dragon-root>/examples/caffe/scripts
mkdir -p ../results/c3d
mkdir -p ../results/resnet
sudo su
./run --log run-c3d.log c3d <path-to-converted-ucf101-data> ../results/c3d
./run --log run-resnet.log resnet <path-to-converted-ilsvrc12-data> ../results/resnet
mv ../results/c3d/result-cpu.data ../results/c3d/result-cpu-atlas.data
mv ../results/resnet/result-cpu.data ../results/resnet/result-cpu-atlas.data
```

5. Recompile Caffe with *OpenBLAS OpenMP*. Please follow the official
[instruction](http://caffe.berkeleyvision.org/installation.html). On CentOS7,
you need to install ```sudo yum install openblas-openmp64.x86_64```.

6. Rerun the experiments with OpenBLAS. This step need root privilege.

```
cd <dragon-root>/examples/caffe/scripts
sudo su
./run --log run-c3d.log --prog cpu c3d <path-to-converted-ucf101-data> ../results/c3d
./run --log run-resnet.log --prog cpu resnet <path-to-converted-ilsvrc12-data> ../results/resnet
mv ../results/c3d/result-cpu.data ../results/c3d/result-cpu-omp.data
mv ../results/resnet/result-cpu.data ../results/resnet/result-cpu-omp.data
mv ../results/c3d/result-cpu-atlas.data ../results/c3d/result-cpu.data
mv ../results/resnet/result-cpu-atlas.data ../results/resnet/result-cpu.data
```

7. Generate Figure 7.

```
cd <dragon-root>/examples/caffe/analyzers
python ptc_resnet_c3d.py    # Figure 7
```
