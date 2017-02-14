#include "genresult.cuh"
#include <sys/time.h>

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

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate here...*/

		/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernels...*/

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
    /*Deallocate.*/
}
