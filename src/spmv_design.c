#include "genresult.cuh"
#include <sys/time.h>

__global__ void designer_kernel(const int M, const int nnz, const int* csr_row, const int* coord_col, const float* A, const float* x, float* y){
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;
	int iter = M % thread_num ? M/thread_num + 1: M/thread_num;

	for(int i = 0 ; i < iter ; i++){
		int datarow = thread_id + (i * thread_num);
		if( datarow < M){			

			int startIdx = csr_row[datarow];
			int endIdx = csr_row[datarow+1];
			
			double tmp = 0;
	
			for(int j = startIdx; j < endIdx; j++){
				float data = A[j];
				int col = coord_col[j];
				tmp = fma( (double)  data, (double) x[col], tmp);
			}
			y[datarow] = tmp;
		}
	}

}

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
	int nnz = mat->nz;
	int numRows = mat->M;
	int numCols = mat->N;
	int *csr_row, *coord_col;
	float *A, *x, *y;

	cudaMalloc((void**)&csr_row, (size_t)(numRows+1)*sizeof(int));
	cudaMalloc((void**)&coord_col, (size_t)nnz*sizeof(int));
	cudaMalloc((void**)&A, (size_t)nnz*sizeof(float));
	cudaMalloc((void**)&x, (size_t)numCols*sizeof(float));
	cudaMalloc((void**)&y, (size_t)numRows*sizeof(float));

	cudaMemcpy(csr_row, mat->rIndex, (size_t)(numRows+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(coord_col, mat->cIndex, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(A, mat->val, (size_t)nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(x, vec->val, (size_t)numCols*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(y, 0, (size_t)numRows*sizeof(float));

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	designer_kernel<<<blockNum, blockSize>>>(numRows, nnz, csr_row, coord_col, A, x, y);

	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Your Own Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

	cudaMemcpy(res->val, y, (size_t)numRows*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(csr_row);
	cudaFree(coord_col);
	cudaFree(A);
	cudaFree(x);
	cudaFree(y);
}
