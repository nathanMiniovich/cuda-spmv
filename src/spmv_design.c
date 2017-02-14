#include "genresult.cuh"
#include <sys/time.h>

/* Put your own kernel(s) here*/

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate*/

    struct timespec start, end;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Your own magic here!*/

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Your Own Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    /*Deallocate*/
}
