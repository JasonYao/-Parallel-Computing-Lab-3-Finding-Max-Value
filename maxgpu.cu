#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <stdbool.h>
#include <math.h>

/* By Jason Yao */
bool IS_DEBUG_MODE = true;
bool IS_SEQUENTIAL_MODE = false;

/**
 * Global definitions
 */
#define THREADS_PER_BLOCK 1024      // Number of threads per block
#define THREADS_PER_SM 2048         // Number of threads per SM

/**
 * Function declarations
 */
void printDeviceInfo();
long getmax(long *, long);
long getmaxcu(long*, long, long);
long getmaxcu2(long*, long, long);
int BLOCKSIZE;

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

    printf("The maximum number in the array is (Sequential): %ld\n", ans);

    if (!IS_SEQUENTIAL_MODE)
        printf("The maximum number in the array is (Parallel): %ld\n", getmaxcu2(numbers, size, ans));

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





//// Kernel that executes on the CUDA device
//__global__ void findMaxCUDA(long* num_device, long size, long* max_device, dim3 threads)
//{
//    __shared__ long max_shared;
//    extern __shared__ long num_shared[];
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;        // Global thread ID
//    int tid = threadIdx.x;                                  // Local thread ID
//
//    num_shared[tid] = INT32_MIN;
//
//    for (long current_global_id = idx; current_global_id < size;)
//    {
//        if (num_shared[tid] > num_device[current_global_id])
//            num_shared[tid] = num_shared[tid];
//        else
//            num_shared[tid] = num_device[current_global_id];
//        current_global_id += gridDim.x*blockDim.x;
//    }
//    __syncthreads();
//
//    for (int i = blockDim.x/2; i > 0; i >>=1)
//    {
//        if ((tid < i) && (idx < size))
//        {
//            if (num_shared[tid] > num_shared[tid + i])
//                num_shared[tid] = num_shared[tid];
//            else
//                num_shared[tid] = num_shared[tid + i];
//        }
//        __syncthreads();
//    }
//
//    if (tid == 0)
//    {
//        max_shared = num_shared[0];
//        for (int i = 0; i < 32; ++i)
//            if (max_shared < num_shared[i])
//                max_shared = num_shared[i];
//        *max_device = max_shared;
//    }
//} // End of the findmax CUDA kernal

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

__global__ void findMaxCUDAOptimised(long* num_device, long size)
{

}

long getmaxcu2(long num[], long size, long ans)
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




//
//
//    cudaDeviceSynchronize();


    // Kernal invocation
    //findMaxCUDA<<<1, 1>>>(num_device, num_size, max_device); // TODO remove after
//    findMaxCUDA<<< grid, threads, BLOCKSIZE>>>(num_device, size, max_device, threads); // Working with old version
//    findMaxCUDAUnoptimised<<< grid, threads, BLOCKSIZE>>>(num_device, unoptimisedSize, max_device);



    findMaxCUDAUnoptimised<<<1, 1, 0>>>(num_device, size);
//    findMaxCUDAOptimised<<<number_of_elements_per_thread, number_of_threads>>>(num_device, size);
//    findMaxCUDAUnoptimised<<< 1, 1, 0>>>(num_device, num_device_size, max_device);


    // Transfers the updated num array back to the host
    long* new_num = (long *)malloc(size * sizeof(long));
    cudaMemcpy(new_num, num_device, sizeof(long) * size, cudaMemcpyDeviceToHost);

    // Iterates through each block to find the max
    long max_value = new_num[0];
//    int i;
//    for (i = 0; i < size; i += threads_per_block)
//    {
//        if (max_value < new_num[i])
//            max_value = num[i];
//    }

//    // Transfer max from device to host
//    long best_ans;
//    cudaMemcpy(&best_ans, max_device, sizeof(long), cudaMemcpyDeviceToHost);
//    cudaMemcpy((void *) num_host, (const void *) num_device, sizeof(long) * size, cudaMemcpyDeviceToHost);
    //max_host = ans;

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










































///*
//   input: pointer to an array of long int
//          number of elements in the array
//   output: the maximum number of the array
//*/
//long getmaxcu(long num[], long size, long ans)
//{
//    // GPU setup
//    int devID = 0;
//    cudaError_t error;
//    struct cudaDeviceProp deviceProp;
//    error = cudaGetDevice(&devID);
//
//    // Error handling
//    if (error != cudaSuccess)
//        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
//
//    error = cudaGetDeviceProperties(&deviceProp, devID);
//
//    if (deviceProp.computeMode == cudaComputeModeProhibited)
//    {
//        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
//        exit(EXIT_SUCCESS);
//    }
//
//    if (error != cudaSuccess)
//        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
//    else
//    {
//        if (IS_DEBUG_MODE)
//            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
//    }
//
//    // Host memory for num[] and size have been allocated and initialised in main()
//    long max_host = num[0];
//    long* num_host = (long *)malloc(size * sizeof(long));
//
//    // Allocate device memory
//    long max_device;
//    long* num_device;
//    unsigned long num_size = sizeof(long) * size;
//    cudaMalloc((void **) &max_device, sizeof(long));
//    cudaMalloc((void **) &num_device, num_size);
//
//    // Transfers num[] to device memory
//    cudaMemcpy(num_device, num, num_size, cudaMemcpyHostToDevice);
//
//    // Sets up execution parameters
//    // Use a larger block size for Fermi and above
//    BLOCKSIZE = (deviceProp.major < 2) ? 16 : 32;
//
//    dim3 dimsA(5*2*BLOCKSIZE, 5*2*BLOCKSIZE, 1);
//    dim3 dimsB(5*4*BLOCKSIZE, 5*2*BLOCKSIZE, 1);
//
//    // Setup execution parameters
//    dim3 threads(BLOCKSIZE, BLOCKSIZE);
//    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
//
//    cudaDeviceSynchronize();
//
//    // Kernal invocation
//    //findMaxCUDA<<<1, 1>>>(num_device, num_size, max_device); // TODO remove after
////    findMaxCUDA<<< grid, threads, BLOCKSIZE>>>(num_device, size, max_device, threads); // Working with old version
////    findMaxCUDAUnoptimised<<< grid, threads, BLOCKSIZE>>>(num_device, unoptimisedSize, max_device);
//    findMaxCUDAUnoptimised<<< 1, 1, 0>>>(num_device, size, max_device);
//
////    if (block_size == 16)
////    {
////        findMaxCUDA<16><<< grid, threads >>>(num_device, num_size, max_device);
////    }
////    else
////    {
////        findMaxCUDA<32><<< grid, threads >>>(num_device, num_size, max_device);
////    }
//
//    // Transfer max from device to host
//    long best_ans;
//    cudaMemcpy((void *) &best_ans, (const void *) max_device, sizeof(long), cudaMemcpyDeviceToHost);
////    cudaMemcpy((void *) num_host, (const void *) num_device, sizeof(long) * size, cudaMemcpyDeviceToHost);
//    //max_host = ans;
//    printf("best_ans: %ld\n", best_ans);
//    max_host = best_ans;
//
//    // Frees all device memory
//    cudaFree((void *) max_device);
//    cudaFree(num_device);
//
//    // cudaDeviceReset causes the driver to clean up all state. While
//    // not mandatory in normal operation, it is good practice.  It is also
//    // needed to ensure correct operation when the application is being
//    // profiled. Calling cudaDeviceReset causes all profile data to be
//    // flushed before the application exits
//    cudaDeviceReset();
//
//    return max_host;
//} // End of the getmaxcu function
















/////////////////////////////////////////////////// DEAD CODE


//    findMaxCUDA<<< grid, threads>>>(num_device, size); // Working with new version
//long* newSize = (long *) size;
//maxNum<<<grid, threads, BLOCKSIZE>>>(num_device, newSize, max_device); // not working




///*
//   input: pointer to an array of long int
//          number of elements in the array
//   output: the maximum number of the array
//*/
//__global__ static void maxNum(long* num, long* size, long* result)
//{
//    long tmp = 0;
//    long i;
//    for (i = 0; i < *size; i++)
//        if (num[i] > tmp)
//            tmp = num[i];
//
//    *result = tmp;
//}

//__global__ void findMaxCUDA(long num[], long size) {
//    long curMax;
//    int index = threadIdx.x + (blockDim.x * blockIdx.x);
//    long count = size;
//
//    while(count > 1)
//    {
//        long halfSize = count / 2;
//        if (index < halfSize){
//            curMax = num[ index + halfSize ];
//            if (curMax > num[ index ]) {
//                num[index] = curMax;
//            }
//        }
//        __syncthreads();
//        count = count / 2;
//    }
//    //cudaGetDeviceProperties()
//}

//
//



// START OF DIMENSION CALCULATION DEAD CODE
//
//
//    // Sets up execution parameters
//    // Use a larger block size for Fermi and above
//    BLOCKSIZE = (deviceProp.major < 2) ? 16 : 32;
//
//    dim3 dimsA(5*2*BLOCKSIZE, 5*2*BLOCKSIZE, 1);
//    dim3 dimsB(5*4*BLOCKSIZE, 5*2*BLOCKSIZE, 1);
//
//    // Setup execution parameters
//    dim3 threads(BLOCKSIZE, BLOCKSIZE);
//    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
//
// END OF DIMENSION CALCULATION DEAD CODE






















