OPS_SRC := $(wildcard ops/*.cpp ops/*.cu)
OPS_OBJ := $(OPS_SRC:%=%.o)

MAIN_SRC := $(wildcard *.cpp *.cu)
MAIN_OBJ := $(MAIN_SRC:%=%.o)

.SUFFIXES:

ops/%.o: ops/% $(wildcard ops/*.h)
	nvcc -o $@ -c $<

%.o: % $(wildcard ops/*.h *.h)
	nvcc -o $@ -c $<

	
main: $(OPS_OBJ) $(MAIN_OBJ)
	nvcc -o $@ $^

all: main

clean:
	rm -rf ops/*.o *.o main
