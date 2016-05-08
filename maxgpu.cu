#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <stdbool.h>
#include <math.h>

/* By Jason Yao */
bool IS_DEBUG_MODE = false;
bool IS_SEQUENTIAL_MODE = false;

/**
 * Global definitions
 */
#define THREADS_PER_BLOCK 1024      // Number of threads per block, hardcoded since all have compute capability of 3.x+

/**
 * Function declarations
 */
void printDeviceInfo();
long getmax(long *, long);
long getmaxcu(long*, long, long);

/* Setup functions */
int main(int argc, char *argv[])
{
    long size = 0;  // The size of the array
    long i;  // loop index
    long * numbers; //pointer to the array

    if(argc != 2)
    {
        printf("usage: maxseq num\n");
        printf("num = size of the array\n");
        exit(1);
    }

    size = atol(argv[1]);

    numbers = (long *)malloc(size * sizeof(long));
    if(!numbers)
    {
        printf("Unable to allocate mem for an array of size %ld\n", size);
        exit(1);
    }

    srand((unsigned int) time(NULL)); // setting a seed for the random number generator

    // Fill-up the array with random numbers from 0 to size-1
    for( i = 0; i < size; i++)
        numbers[i] = rand() % size;

    if (IS_DEBUG_MODE)
    {
        printf("The data array:\n");
        for( i = 0; i < size; i++)
            printf("%ld\n", numbers[i]);
        printDeviceInfo();
    }

    long ans = getmax(numbers, size);

    if (!IS_SEQUENTIAL_MODE)
        printf("The maximum number in the array is: %ld\n", getmaxcu(numbers, size, ans));

    free(numbers);
    return EXIT_SUCCESS;
} // End of the main function

void printDeviceInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++)
    {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
} // End of the print device info function

/*
 * Sequential version to find the maximum of
 * an array of long integers.
*/
long getmax(long num[], long size)
{
    long i;
    long max = num[0];

    for(i = 1; i < size; i++)
        if(num[i] > max)
            max = num[i];

    return( max );
} // End of the sequential getmax function

/**
 * Unoptimised kernal to find the maximum value of an array of longs
 */
__global__ void findMaxCUDAUnoptimised(long* num_device, long size)
{
    long i;
    for (i = 0; i < size; ++i)
    {
        if (num_device[i] > num_device[0])
            num_device[0] = num_device[i];
    }
}

/**
 * Optimised kernal to find the maximum value of an array of longs
 */
__global__ void findMaxCUDAOptimised(long* num_device, long size, long* max_of_each_block_device)
{
    extern __shared__ long num_shared[];
    int thread_id = threadIdx.x;                        // Local thread id inside block
    int idx = blockDim.x * blockIdx.x + threadIdx.x;    // Global unique thread id in grid

    // Grabs from global memory to shared memory if available
    num_shared[thread_id] = 0;
    if (idx < size)
        num_shared[thread_id] = num_device[idx];
    __syncthreads();

    max_of_each_block_device[thread_id] = num_shared[idx];

    // Iterates through the shared memory version of the num[] to find the largest
    for (unsigned int s = blockDim.x; s > 1; s >>= 1)
    {
        int halfway_point = s >> 1;
        if (threadIdx.x < halfway_point)
        {
            long temp = max_of_each_block_device[threadIdx.x + halfway_point];
            if (temp > max_of_each_block_device[threadIdx.x])
                max_of_each_block_device[threadIdx.x] = temp;
        }
        __syncthreads();
    }

    if (thread_id == 0)
        printf("\nMax element on GPU=%ld\n", max_of_each_block_device[0]);
}

long getmaxcu(long num[], long size, long ans)
{
    // GPU setup
    int devID = 0;
    cudaError_t error;
    struct cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    // Error handling
    if (error != cudaSuccess)
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    else
    {
        if (IS_DEBUG_MODE)
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Host memory for num[] and size have been allocated and initialised in main()

    // Calculates the dimensions
    long number_of_elements_per_thread = (long) ceil(size / THREADS_PER_BLOCK);
    long max_value = ans;
    long threads_per_block;
    if (size > THREADS_PER_BLOCK)
        threads_per_block = THREADS_PER_BLOCK;
    else
        threads_per_block = size;
    long blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // Allocates and copies the whole array to the device
    long* num_device;
    cudaMalloc((void**) &num_device, sizeof(long) * size);
    cudaMemcpy((void*) num_device, (void*) num, sizeof(long) * size, cudaMemcpyHostToDevice);

    long* max_of_each_block_device;
    cudaMalloc((void**) &max_of_each_block_device, sizeof(long) * blocks_per_grid);

    // Kernal invocation
    cudaDeviceSynchronize();
    findMaxCUDAUnoptimised<<<1, 1>>>(num_device, size);                                 // Unoptimised call
//    findMaxCUDAOptimised<<<1, 1, 0>>>(num_device, size, max_of_each_block_device);    // Optimised call

    // Transfers the output array to the host
    long* new_num = (long *)malloc(sizeof(long) * blocks_per_grid);
    cudaMemcpy(new_num, max_of_each_block_device, sizeof(long) * blocks_per_grid, cudaMemcpyDeviceToHost);

    // Frees all device memory
    cudaFree(num_device);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    return max_value;
} // End of the getmaxcu function