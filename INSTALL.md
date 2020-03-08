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
* NVIDIA GPU driver 
  * Version 384.81 
    * compatible with CUDA 9.0 and below
    * compatible with linux kernel version 3.10 or above but below 4.0
    * tested on CentOS 7
  * Version 410.48 
    * compatible with CUDA 9.0 and below
    * compatible with linux kernel version 3.10 or above but below 4.0
    * tested on CentOS 7
  * Version 440.33.01 
    * compatible with CUDA 10.2 and below
    * compatible linux kernel version 5.0 or above
    * tested on Ubuntu 18.04 (linux kernel 5.3.0-40-generic)

### Runtime environment

* Linux OS (tested on CentOS 7 and Ubuntu 18.04)
* Linux kernel version 3.10 or above but below 4.0, or 5.0 and above (incompatible with kernel 4.x)
* Storage formatted with the ext4 filesystem

### Hardware

* An NVIDIA Pascal P100 GPU or above that supports GPU hardware page-fault

## How to prepare *dragon-driver*

**Note:** You need to replace *<version>* with the NVIDIA GPU driver version you
choose. Install that driver version on your system first before follow the
instructions below.

1. Get NVIDIA GPU driver version 384.81, 410.48, or 440.33.01 from [the NVIDIA
website](https://www.nvidia.com/drivers/beta). The downloaded driver file should
be in \*.run file format.

2. If the downloaded file is not executable
```
chmod +x NVIDIA-Linux-x86_64-<version>.run
```

3. Extract this file to the *drivers* directory.  After execute this
instruction, the *NVIDIA-Linux-x86_64-\<version\>* directory should appear in the
*drivers* directory.  Due to license issues, we cannot distribute this driver
with this repository.
```
cd <dragon-root>/drivers
./NVIDIA-Linux-x86_64-<version>.run -x
```

4. Run the *prepare-dragon-driver* script in the *scripts* folder.
```
../scripts/prepare-dragon-driver nvidia-uvm-<version>.patch NVIDIA-Linux-x86_64-<version>
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
sudo ./<dragon-root>/scripts/activate-dragon
```
