/*------------------------------------------------------------------------------------------------
local_mpi_to_gpu

Written by Tom Papatheodore
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <sched.h>
#include <mpi.h>

int main(int argc, char *argv[])
{

	/* -------------------------------------------------------------------------------------------
		MPI Initialization 
	--------------------------------------------------------------------------------------------*/
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char name[MPI_MAX_PROCESSOR_NAME];
	int resultlength;
	MPI_Get_processor_name(name, &resultlength);
	
	const char* nl_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
	int node_local_rank = atoi(nl_rank);

	/* -------------------------------------------------------------------------------------------
		Other Initialization 
	--------------------------------------------------------------------------------------------*/

	// Find hardware thread being used by each MPI rank
	int hwthread = sched_getcpu();

	// Find how many GPUs CUDA runtime says are available
	int num_devices = 0;
	cudaGetDeviceCount(&num_devices);

	// Map MPI ranks to GPUs according to node-local MPI rank (round-robin)
	int gpu_id = node_local_rank % num_devices;
	cudaSetDevice(gpu_id);

	// Check which GPU each MPI rank is actually mapped to
	int device;
	cudaGetDevice(&device);

	/* -------------------------------------------------------------------------------------------
		Output and finalize
	--------------------------------------------------------------------------------------------*/

	// Each MPI rank will print its details.
	printf("Global Rank: %03d of %03d, Local Rank: %03d, HWThread %03d, GPU: %01d Node: %s\n", rank, size, node_local_rank, hwthread, device, name);

	MPI_Finalize();

	return 0;
}
