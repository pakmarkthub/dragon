all: cudamemcpy uvm hostreg nvmgpu

cudamemcpy: bin/cudamemcpy

uvm: bin/uvm

hostreg: bin/hostreg

nvmgpu: bin/nvmgpu

bin/cudamemcpy:
	cd cudamemcpy && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../cudamemcpy/bin/vectorAdd bin/cudamemcpy

bin/uvm:
	cd uvm && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../uvm/bin/vectorAdd bin/uvm

bin/hostreg:
	cd hostreg && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../hostreg/bin/vectorAdd bin/hostreg

bin/nvmgpu:
	cd nvmgpu && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../nvmgpu/bin/vectorAdd bin/nvmgpu

bin/ap:
	cd ap && $(MAKE)
	mkdir -p bin
	cd bin
	ln -s ../ap/bin/vectorAdd bin/ap

clean:
	rm -rf bin
	cd cudamemcpy && $(MAKE) clean
	cd uvm && $(MAKE) clean
	cd hostreg && $(MAKE) clean
	cd nvmgpu && $(MAKE) clean
	cd ap && $(MAKE) clean
