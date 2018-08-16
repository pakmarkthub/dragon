all: dragon.c ../include/dragon.h
	gcc -shared -o libdragon.so -fPIC dragon.c -I../include/ -D_GNU_SOURCE -m64 -lcudart -Wall -lglib-2.0 -O3
	mkdir -p ../lib64/
	mv libdragon.so ../lib64/

clean:
	rm ../lib64/*.so
