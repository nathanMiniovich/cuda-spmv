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
	int nnz = mat->nz;
	int M = mat->M;
	int *coord_row, *coord_col;
	float *A, *x, *y;
	cudaMallocManaged((void**)&coord_row, (size_t)nnz*sizeof(int));
	cudaMallocManaged((void**)&coord_col, (size_t)nnz*sizeof(int));
	cudaMallocManaged((void**)&A, (size_t)nnz*sizeof(float));
	cudaMallocManaged((void**)&x, (size_t)M*sizeof(float));
	cudaMallocManaged((void**)&y, (size_t)M*sizeof(float));
	/* Initialize here...*/
	cudaMemcpy(coord_row, mat->rIndex, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(coord_col, mat->cIndex, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(A, mat->val, (size_t)nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(x, vec->val, (size_t)M*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(y, 0, (size_t)M*sizeof(float));
	/* Sample timing code */
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	/*Invoke kernels...*/
	getMulAtomic_kernel<<<blockNum, blockSize>>>(nnz, coord_row, coord_col, A, x, y);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Atomic Kernelll Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
	/* Write back result */
	cudaMemcpy(res->val, y, (size_t)M*sizeof(float), cudaMemcpyDeviceToHost);
	/*Deallocate.*/
	cudaFree(coord_row);
	cudaFree(coord_col);
	cudaFree(A);
	cudaFree(x);
	cudaFree(y);
}
