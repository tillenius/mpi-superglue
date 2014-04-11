ifeq ("x$(MKL_ROOT)x","xx")
$(error MKL_ROOT is not set)
endif

ifeq ("x$(SUPERGLUE_ROOT)x","xx")
$(error SUPERGLUE_ROOT is not set)
endif



EXAMPLES:=$(basename $(notdir $(wildcard src/*.cpp)))

FLAGS=-g -O3 -pedantic -Wall -Wno-long-long -I $(SUPERGLUE_ROOT)/include -I ../include -pthread

MKL_INC=-isystem $(MKL_ROOT)/include
MKL_LIBS= \
  -Wl,--start-group \
  $(MKL_ROOT)/lib/intel64/libmkl_intel_lp64.a \
  $(MKL_ROOT)/lib/intel64/libmkl_sequential.a \
  $(MKL_ROOT)/lib/intel64/libmkl_core.a \
  -ldl \
  -Wl,--end-group

all: $(EXAMPLES)

mpi_cholesky: bin/ src/mpi_cholesky.cpp Makefile
	mpic++ $(FLAGS) $(MKL_INC) src/mpi_cholesky.cpp -o bin/$@ $(MKL_LIBS) -lm

#$(EXAMPLES): bin/
#	mpic++ $(FLAGS) src/$@.cpp -o bin/$@

bin:
	mkdir -p bin

clean:
	rm -rf bin

.PHONY: all clean $(EXAMPLES)

