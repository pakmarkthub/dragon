# Installation Instructions for DRAGON

DRAGON composes of two components:
* **dragon-driver** - modified *nvidia-uvm* driver
* **libdragon** - user-space library for communicating with the modified driver

Both components need to be set up properly in order to use DRAGON.

## Prerequisites

### Software and libraries

* Python2.7 or above
* glibc-2.0
* CUDA version 9.0 or above
* NVIDIA GPU driver version 384.81 -- currently support this version only

### Runtime environment

* Linux OS (tested on CentOS 7)
* Linux kernel version 3.10 or above but below 4.0 (incompatible with kernel 4.x)
* Storage formatted with the ext4 filesystem

### Hardware

* An NVIDIA Pascal P100 GPU or above that supports GPU hardware page-fault

## How to prepare *dragon-driver*

1. Get NVIDIA GPU driver version 384.81 from [the NVIDIA
website](https://www.nvidia.com/drivers/beta). The downloaded driver file should
be in \*.run file format.

2. If the downloaded file is not executable
```
chmod +x NVIDIA-Linux-x86_64-384.81.run
```

3. Extract this file to the *drivers* directory.  After execute this
instruction, the *NVIDIA-Linux-x86_64-384.81* directory should appear in the
*drivers* directory.  Due to license issues, we cannot distribute this driver
with this repository.
```
cd <dragon-root>/drivers
./NVIDIA-Linux-x86_64-384.81.run -x
```

4. Run the *prepare-dragon-driver* script in the *scripts* folder.
```
../scripts/prepare-dragon-driver nvidia-uvm-384.81.patch NVIDIA-Linux-x86_64-384.81
```

## How to compile *libdragon*

```
cd <dragon-root>/library/src
make -j
```

## How to setup DRAGON environment

Up to this step, DRAGON is not permanently installed on your machine. All of the
modified files are contained in this repository. To activate DRAGON, run the
following script. Note that you should run this script again after your machine
has rebooted.
```
./<dragon-root>/scripts/activate-dragon
```
