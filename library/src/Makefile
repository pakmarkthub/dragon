GLIB_CFLAGS ?= $(shell pkg-config --cflags glib-2.0)
GLIB_LIBS ?= $(shell pkg-config --libs glib-2.0)

all: dragon.c ../include/dragon.h
	gcc -shared -o libdragon.so -fPIC dragon.c -I../include/ -D_GNU_SOURCE -m64 -lcudart -Wall $(GLIB_CFLAGS) $(GLIB_LIBS) -O3
	mkdir -p ../lib64/
	mv libdragon.so ../lib64/

clean:
	rm ../lib64/*.so
