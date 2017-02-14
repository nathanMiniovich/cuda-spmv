#include "genresult.cuh"
#include <sys/time.h>


__global__ void putProduct_kernel(/*Arguments*/){
    /*Put your kernel(s) implementation here, you don't have to use exactly the
 * same kernel name */
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate things...*/

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernel(s)*/

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    /*Deallocate, please*/
}
