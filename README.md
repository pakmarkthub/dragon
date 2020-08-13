# DRAGON: Direct Resource Access for GPUs Over NVM

DRAGON is a host-based framework that transparently extends the GPU addressable
global memory space beyond the host memory using NVM-backed data pointers.
DRAGON allows storing the binary memory dump of application data in a file on
NVM and mapping it to the global memory space of the GPU. This enables GPU
kernels to access the data via regular load/store instructions, similar to how
*mmap()* operates in CPUs.

## Contact Us

To people who reach this page from our paper: 

My university email address, which is written on the paper, will be expired
soon. You can contact me via "pak.markthub+dragon at gmail dot com", or by
opening an issue on GitHub.

Pak Markthub

## Getting Started

This project composes of multiple components. To install DRAGON, follow the
instructions in the [INSTALL.md](INSTALL.md) file. To compile or run provided
example applications, see the [README.md](examples/README.md) file in the
*examples* folder.

### Experimental Features

**Move to private repo for now.**

All features are tracked in branch [dev](https://github.com/pakmarkthub/dragon/tree/dev).

1. DRAGON-DIRECT
   - Use DMA to transfer data from NVMe to GPU.
   - No host page caching in between.

2. DRAGON-Resident
   - Query the current physical location on the mapped buffers at runtime.
   - Pin pages in GPU memory. Disable eviction of those pinned pages.

## Versioning

For the versions available, see the [tags on this repository](https://github.com/pakmarkthub/dragon/tags). 

## Authors

* [Pak Markthub](https://www.linkedin.com/in/pakmarkthub) (pak.markthub+dragon at gmail dot com)
* [Mehmet E. Belviranli](https://ft.ornl.gov/~belviranli/)
* [Seyong Lee](https://ft.ornl.gov/~lees2/)
* [Jeffrey S. Vetter](https://ft.ornl.gov/~vetter/)
* [Satoshi Matsuoka](http://www.r-ccs.riken.jp/en/overview/leadership.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This project was partially supported by JST CREST Grant Numbers JPMJCR1303
(EBD CREST) and JPMJCR1687 (DEEP CREST), and performed under the auspices of
Real-World Big-Data Computation Open Innovation Laboratory (RWBC-OIL), Japan.
This project was also supported in part by Oak Ridge National Laboratory ASTRO
Program sponsored by the US Department of Energy and administered by the Oak
Ridge Institute for Science and Education, USA. 

This material is based upon work supported by the U.S. Department of Energy,
Office of Science, Office of Advanced Scientific Computing Research, under
contract number DE-AC05-00OR22725. This manuscript has been authored by
UT-Battelle, LLC, under contract DE-AC05-00OR22725 with the US Department of
Energy (DOE). The US government retains and the publisher, by accepting the
article for publication, acknowledges that the US government retains a
nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce
the published form of this manuscript, or allow others to do so, for US
government purposes. DOE will provide public access to these results of
federally sponsored research in accordance with the DOE Public Access Plan
(http://energy.gov/downloads/doe-public-access-plan).

## Notes

* **nvmgpu** is the former name of this project. We use *nvmgpu* and *dragon*
interchangeably throughout the source code.

* If you want to reproduce the results we reported in our SC18 paper *DRAGON:
Breaking GPU Memory Capacity Limits with Direct NVM Access*, look at the
[README.md](examples/README.md) file in the *examples* folder for the
up-to-dated instructions.

* If you use this work, please cite our paper.

    * Pak Markthub, Mehmet E. Belviranli, Seyong Lee, Jeffrey S. Vetter, and Satoshi
Matsuoka. DRAGON: Breaking GPU Memory Capacity Limits with Direct NVM Access. In
Proceedings of the International Conference for High Performance Computing,
Networking, Storage and Analysis (SC18). ACM, Dallas, USA, November 2018.
