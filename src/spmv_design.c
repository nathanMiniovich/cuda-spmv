#include "genresult.cuh"
#include <sys/time.h>

__global__ void designer_kernel(const int nnz, const int* coord_row, const int* coord_col, const float* A, const float* x, float* y){

}

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
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

	designer_kernel<<<blockNum, blockSize>>>(nnz, coord_row, coord_col, A, x, y);

	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Your Own Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

	cudaMemcpy(res->val, y, (size_t)numRows*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(coord_row);
	cudaFree(coord_col);
	cudaFree(A);
	cudaFree(x);
	cudaFree(y);
}
