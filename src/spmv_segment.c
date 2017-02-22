#include "genresult.cuh"
#include <sys/time.h>
#define MAX_PER_BLOCK 1024

/*
 * segmented scan in shared memory, assuming corresponding A values
 * are loaded into the shared memory array vals, the row indices loaded
 * into rows[] array in shared memory
 * lane is the thread offset in the thread warp 
 */
__device__ void segmented_scan(const int lane, const int *rows, float *vals, float * y){
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

	if ( lane == 31 || rows[threadIdx.x] != rows[threadIdx.x + 1] ){
		atomicAdd(&y[rows[threadIdx.x]], vals[threadIdx.x]);
	}
}

__global__ void putProduct_kernel(const int nnz, const int* coord_row, const int* coord_col, const float* A, const float* x, float* y){
	extern __shared__ int rows[MAX_PER_BLOCK];
	extern __shared__ float vals[MAX_PER_BLOCK];

        int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        int thread_num = blockDim.x * gridDim.x;
        int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;

        for(int i = 0; i < iter; i++){
                int dataid = thread_id + i * thread_num;

                if(dataid < nnz){
                        float data = A[dataid];
			rows[threadIdx.x] = coord_row[dataid];
                        int col = coord_col[dataid];
                        vals[threadIdx.x] = data * x[col];
			// __syncthreads();
			segmented_scan(thread_num, iter, thread_id % 32, rows, vals, y);
                }
        }
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
	int nnz = mat->nz;
        int numRows = mat->M;
	int numCols = mat->N;
        int *coord_row, *coord_col;
        float *A, *x, *y;

        cudaMalloc((void**)&coord_row, (size_t)nnz*sizeof(int));
        cudaMalloc((void**)&coord_col, (size_t)nnz*sizeof(int));
        cudaMalloc((void**)&A, (size_t)nnz*sizeof(float));
        cudaMalloc((void**)&x, (size_t)numCols*sizeof(float));
        cudaMalloc((void**)&y, (size_t)numRows*sizeof(float));

        cudaMemcpy(coord_row, mat->rIndex, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(coord_col, mat->cIndex, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(A, mat->val, (size_t)nnz*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(x, vec->val, (size_t)numCols*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(y, 0, (size_t)numRows*sizeof(float));
	
    	struct timespec start, end;
    	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	putProduct_kernel<<<blockNum, blockSize>>>(nnz, coord_row, coord_col, A, x, y);

    	cudaDeviceSynchronize();
    	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    	printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

	cudaMemcpy(res->val, y, (size_t)numRows*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(coord_row);
        cudaFree(coord_col);
        cudaFree(A);
        cudaFree(x);
        cudaFree(y);
}
