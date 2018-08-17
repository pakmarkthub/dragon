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
* CUDA version 9.0 or above
* DRAGON library and driver

### Compiling

```
cd <application>/programs
make -j
```

### How to use

This section provides steps for reproducing the results shown in our paper.

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
