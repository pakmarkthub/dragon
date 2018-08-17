all: cudamemcpy uvm hostreg nvmgpu generator

cudamemcpy: bin/cudamemcpy

uvm: bin/uvm

hostreg: bin/hostreg

nvmgpu: bin/nvmgpu

generator: bin/generator

bin/cudamemcpy:
	cd cudamemcpy && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../cudamemcpy/bin/srad bin/cudamemcpy

bin/uvm:
	cd uvm && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../uvm/bin/srad bin/uvm

bin/hostreg:
	cd hostreg && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../hostreg/bin/srad bin/hostreg

bin/nvmgpu:
	cd nvmgpu && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../nvmgpu/bin/srad bin/nvmgpu

bin/generator:
	cd cudamemcpy && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../cudamemcpy/bin/generator bin/generator

clean:
	rm -rf bin
	cd cudamemcpy && $(MAKE) clean
	cd hostreg && $(MAKE) clean
	cd uvm && $(MAKE) clean
	cd nvmgpu && $(MAKE) clean
