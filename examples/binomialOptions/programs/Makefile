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
	ln -s ../cudamemcpy/bin/binomialOptions bin/cudamemcpy

bin/uvm:
	cd uvm && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../uvm/bin/binomialOptions bin/uvm

bin/hostreg:
	cd hostreg && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../hostreg/bin/binomialOptions bin/hostreg

bin/nvmgpu:
	cd nvmgpu && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../nvmgpu/bin/binomialOptions bin/nvmgpu

bin/generator:
	cd generator && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../generator/bin/binomialOptions bin/generator

clean:
	rm -rf bin
	cd cudamemcpy && $(MAKE) clean
	cd uvm && $(MAKE) clean
	cd hostreg && $(MAKE) clean
	cd nvmgpu && $(MAKE) clean
	cd generator && $(MAKE) clean
