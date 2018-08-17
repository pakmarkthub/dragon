FLAGS := -Wno-write-strings

all: hotspotex

hotspotex: hotspotex.cpp
	mkdir -p bin
	g++ $(FLAGS) -o bin/$@ $<

clean:
	rm bin/*

