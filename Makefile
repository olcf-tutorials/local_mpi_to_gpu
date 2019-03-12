CUCOMP  = nvcc
CUFLAGS = -arch=sm_70

INCLUDES  = -I$(OMPI_DIR)/include
LIBRARIES = -L$(OMPI_DIR)/lib -lmpi_ibm

local_mpi_to_gpu: local_mpi_to_gpu.o
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) local_mpi_to_gpu.o -o local_mpi_to_gpu

local_mpi_to_gpu.o: local_mpi_to_gpu.cu
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c local_mpi_to_gpu.cu

.PHONY: clean

clean:
	rm -f local_mpi_to_gpu *.o
