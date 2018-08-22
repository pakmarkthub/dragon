all: cudamemcpy uvm hostreg nvmgpu generator ap

cudamemcpy: bin/cudamemcpy

uvm: bin/uvm

hostreg: bin/hostreg

nvmgpu: bin/nvmgpu

ap: bin/ap

generator: bin/generator

bin/cudamemcpy:
	cd cudamemcpy && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../cudamemcpy/bin/BlackScholes bin/cudamemcpy

bin/uvm:
	cd uvm && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../uvm/bin/BlackScholes bin/uvm

bin/hostreg:
	cd hostreg && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../hostreg/bin/BlackScholes bin/hostreg

bin/nvmgpu:
	cd nvmgpu && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../nvmgpu/bin/BlackScholes bin/nvmgpu

bin/ap:
	cd ap && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../ap/bin/BlackScholes bin/ap

bin/generator:
	cd cudamemcpy && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../cudamemcpy/bin/generator bin/generator

clean:
	rm -rf bin
	cd cudamemcpy && $(MAKE) clean
	cd uvm && $(MAKE) clean
	cd hostreg && $(MAKE) clean
	cd nvmgpu && $(MAKE) clean
	cd ap && $(MAKE) clean
