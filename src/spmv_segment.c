#include "genresult.cuh"
#include <sys/time.h>

__device__ void segmented_scan(const int lane, const int *rows, float *vals){
	// segmented scan in shared memory, assuming corresponding A values
	// are loaded into the shared memory array vals, the row indices loaded
	// into rows[] array in shared memory
	// lane is the thread offset in the thread warp
	
	if ( lane >= 1 && rows[threadIdx.x] == rows[threadIdx.x - 1] )
		vals[threadIdx.x] += vals[threadIdx.x - 1];
	if ( lane >= 2 && rows[threadIdx.x] == rows[threadIdx.x - 2] )
		vals[threadIdx.x] += vals[threadIdx.x - 2];
	if ( lane >=4 && rows[threadIdx.x] == rows[threadIdx.x - 4] )
		vals[threadIdx.x] += vals[threadIdx.x - 4];
	if ( lane >= 8 && rows[threadIdx.x] == rows[threadIdx.x - 8] )
		vals[threadIdx.x] += vals[threadIdx.x - 8];
	if ( lane >= 16 && rows[threadIdx.x] == rows[threadIdx.x - 16] )
		vals[threadIdx.x] += vals[threadIdx.x - 16];
}

__global__ void putProduct_kernel(/*Arguments*/){
    /*Put your kernel(s) implementation here, you don't have to use exactly the
 * same kernel name */
}

/*
__global__ void getMulAtomic_kernel(const int nnz, const int* coord_row, const int* coord_col, const float* A, const float* x, float* y){
        int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        int thread_num = blockDim.x * gridDim.x;
        int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;

        for(int i = 0; i < iter; i++){
                int dataid = thread_id + i * thread_num;
                if(dataid < nnz){
                        float data = A[dataid];
                        int row = coord_row[dataid];
                        int col = coord_col[dataid];
                        float tmp = data * x[col];
                        atomicAdd(&y[row], tmp);
                }
        }
}
*/


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
