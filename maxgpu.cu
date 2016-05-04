#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>

/* By Jason Yao */
//typedef enum { false, true } bool;      /* Enables boolean types */

/**
 * Function declarations
 */
//long getmax(long *, long);
//long getmaxcu(long num[], long size);


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

    for( i = 0; i < size; i++)
        printf("%ld\n", numbers[i]);

    printDeviceInfo();

//    printf("The maximum number in the array is (GPU): %ld\n",
//           getmaxcu(numbers, size));
//    printf("This compiled corectly");

    free(numbers);
    exit(EXIT_SUCCESS);
} // End of the main function

/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
//long getmaxcu(long num[], long size)
//{
//    long i;
//    long max = num[0];
//    long size_max = sizeof(long);
//    long num_d[] = (long *)malloc(size * sizeof(long));
//
//    // Transfers num[] to device memory
//    cudaMalloc((void **) &num, size);
//    cudaMemcpy(num_d, num, size, cudaMemcpyHostToDevice);
//
//    // Allocate device memory for answer
//    cudaMalloc((void **) &max, size_max);
//
//    // Kernel invocation code TODO
//    //dim3 dimGrid();
//
////    vecAddKernel<<<ceil(n/256)>>>();
//
//    // Transfer max from device to host
//    cudaMemcpy(num, num_d, size, cudaMemcpyDeviceToHost);
//
//    // Frees device memory for num_d and max
//    cudaFree(num);
//    cudaFree(&max);
//    free(num_d);
//
////    free(num_d);
////    for(i = 1; i < size; i++)
////        if(num[i] > max)
////            max = num[i];
//
//  return(max);
//} // End of the getmaxcu function
//
//__global__ long findLongKernal()
//{
//
//}
