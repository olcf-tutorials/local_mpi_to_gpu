# Use node-local MPI rank IDs to map MPI ranks to GPUs

In this tutorial, we will demonstrate how to use node-local MPI rank IDs to map MPI ranks to GPUs. To do so, we will use a simple program called `local_mpi_to_gpu`. 

Before getting started, consider the following example to understand what we mean by "node-local" MPI rank ID and also to motivate why this might be useful:

Assume you have 24 total MPI ranks spread over 4 compute nodes (with 6 ranks on each node). Then, globally, they would be numbered 0-23. However, their node-local rank IDs would be 0-5 on each node, which might be easier to deal with if you need to manually map MPI ranks to GPUs.

Now let's walk through the code to see how it works:

```c
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
```

Just inside the `main` function, we initialize the MPI context, determine the total number of MPI ranks, determine the (global) ID for each MPI rank, find the hostname of the node that each MPI rank is running on, and determine the node-local ID for each MPI rank. For the last part, we simply read the `OMPI_COMM_WORLD_LOCAL_RANK` environment variable and convert the string to an integer (since it will be used as an integer later). 

```c
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
```

In the "Other Intialization" section of the code, we determine the hardware thread ID that each MPI rank is running on, determine how many GPUs are available to each MPI rank, and map the MPI ranks to GPUs according to each rank's node-local ID. Here we use the modulus operator `%` to map ranks to GPUs in a round-robin fashion. Although we could use the variable `gpu_id` to print out the GPU ID that each rank is mapped to, we include the call to `cudaGetDevice` just to make sure the mapping is correct.

```c
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
```

And, finally, we print out the global ID, node-local ID, hardware thread ID, GPU ID, and node hostname  associated with each MPI rank, and close the MPI context.

```c
    /* -------------------------------------------------------------------------------------------
        Output and finalize
    --------------------------------------------------------------------------------------------*/

    // Each MPI rank will print its details.
    printf("Global Rank: %03d of %03d, Local Rank: %03d, HWThread %03d, GPU: %01d Node: %s\n", rank, size, node_local_rank, hwthread, device, name);

    MPI_Finalize();
```

## Compiling and running on Summit
To run this program on Summit, you must first load the CUDA module:

```
$ module load cuda
```

Now compile the program

```
$ make
```

Now submit the job using the `submit.lsf` batch script (make sure to change `PROJ123` to a project you are associated with):

```
$ bsub submit.lsf
```

The `jsrun` line in this script will launch 14 total MPI ranks (7 on each of the 2 nodes). You can check the status of your job with the `jobstat` command. Once your job has completed, you can view the results in the output file named `mpi_ranks.JOBID`:

```
$ cat map_ranks.JOBID

Global Rank: 000 of 014, Local Rank: 000, HWThread 000, GPU: 0 Node: h11n08
Global Rank: 001 of 014, Local Rank: 001, HWThread 005, GPU: 1 Node: h11n08
Global Rank: 002 of 014, Local Rank: 002, HWThread 008, GPU: 2 Node: h11n08
Global Rank: 003 of 014, Local Rank: 003, HWThread 012, GPU: 3 Node: h11n08
Global Rank: 004 of 014, Local Rank: 004, HWThread 017, GPU: 4 Node: h11n08
Global Rank: 005 of 014, Local Rank: 005, HWThread 021, GPU: 5 Node: h11n08
Global Rank: 006 of 014, Local Rank: 006, HWThread 026, GPU: 0 Node: h11n08
Global Rank: 007 of 014, Local Rank: 000, HWThread 001, GPU: 0 Node: h11n13
Global Rank: 008 of 014, Local Rank: 001, HWThread 007, GPU: 1 Node: h11n13
Global Rank: 009 of 014, Local Rank: 002, HWThread 009, GPU: 2 Node: h11n13
Global Rank: 010 of 014, Local Rank: 003, HWThread 014, GPU: 3 Node: h11n13
Global Rank: 011 of 014, Local Rank: 004, HWThread 017, GPU: 4 Node: h11n13
Global Rank: 012 of 014, Local Rank: 005, HWThread 020, GPU: 5 Node: h11n13
Global Rank: 013 of 014, Local Rank: 006, HWThread 026, GPU: 0 Node: h11n13

...

```

Notice that there are 14 total MPI ranks (numbered 0-13), with 7 ranks on each of the 2 compute nodes (h11n08 and h11n13), but their node-local MPI rank IDs are 0-6 within each node. Also notice that local MPI rank IDs 0-5 get mapped to GPUs 0-5, but local MPI rank ID 6 gets mapped to GPU 0; this is because of the round-robin fashion that we mapped ranks to GPUs. 

>NOTE: This simple example only shows how to use the node-local MPI rank IDs to map ranks to GPUs. However, the *mapping of MPI ranks to CPU cores* used in this example is not very efficient (since all MPI ranks are associated with CPU cores on the first socket of a node, while the GPUs those MPI ranks map to are on both sockets - i.e., the MPI ranks mapped to GPUs 3, 4, and 5 must reach across to the second socket to communicate with their GPUs). How you choose to map MPI ranks to CPU cores (defined via `jsrun`) will be specific to each application. For the example above, a better approach might be to create 2 resource sets per node (each consisting of the CPUs and GPUs on a socket - i.e., 21 physical CPU cores and 3 GPUs). That way, the MPI ranks on the first socket only have access to the GPUs on the first socket.

>NOTE: GPUs IDs (as labeled by the CUDA runtime) start over at 0 within each resource set. So defining 2 resource sets per node (where each resource set consists of the CPUs and GPUs on a socket) would show the GPUs in the first resource set labeled as 0-2 and also the GPUs in the second resource set would be labeled as 0-2 (even though the GPUs in the second resource set would actually be GPUs 3-5 as shown on the node diagrams).

>NOTE: When multiple MPI ranks will access the same GPU, the <a href="https://www.olcf.ornl.gov/for-users/system-user-guides/summit/summit-user-guide/#volta-multi-process-service">CUDA Multi-Process Server (MPS)</a> should be started using `-alloc_flags gpumps` in your batch script.

## Helpful Links

CUDA C Programming Model: <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model">https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model</a>

## Problems?
If you see a problem with the code or have suggestions to improve it, feel free to open an issue.


