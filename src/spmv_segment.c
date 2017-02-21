#include "genresult.cuh"
#include <sys/time.h>

__device__ void segmented_scan(const int lane, const int *rows, float *vals, float * y){
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

__global__ void putProduct_kernel(const int  N, const int nnz, const int* coord_row, const int* coord_col, const float* A, const float* x, float* y){
	// dynamically allocated mem 
	// rows - smem[c] 			(length nnz)
	// vals - smem[nnz + c] 		(length nnz)
	extern __shared__ smem[];

        int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        int thread_num = blockDim.x * gridDim.x;
        int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;

        for(int i = 0; i < iter; i++){
                int dataid = thread_id + i * thread_num;

		smem[dataid] = coord_row[dataid];

                if(dataid < nnz){
                        float data = A[dataid];
                        int row = coord_row[dataid];
                        int col = coord_col[dataid];
                        smem[nnz + dataid] = data * x[col];
                }
        }
	__syncthreads();
	segmented_scan(thread_idx % 32, smem, smem + nnz, y);
	
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate things...*/
	int nnz = mat->nz;
        int M = mat->M;
	int N = mat->N;
        int *coord_row, *coord_col;
        float *A, *x, *y;

        cudaMalloc((void**)&coord_row, (size_t)nnz*sizeof(int));
        cudaMalloc((void**)&coord_col, (size_t)nnz*sizeof(int));
        cudaMalloc((void**)&A, (size_t)nnz*sizeof(float));
        cudaMalloc((void**)&x, (size_t)M*sizeof(float));
        cudaMalloc((void**)&y, (size_t)M*sizeof(float));

        cudaMemcpy(coord_row, mat->rIndex, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(coord_col, mat->cIndex, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(A, mat->val, (size_t)nnz*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(x, vec->val, (size_t)M*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(y, 0, (size_t)M*sizeof(float));
	
	// init vals and rows	
    	struct timespec start, end;
    	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    	/*Invoke kernel(s)*/
	putProduct_kernel<<<blockNum, blockSize, 2*nnz>>>(N, nnz, coord_row, coord_col, A, x, y);
    	cudaDeviceSynchronize();
    	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    	printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    	/*Deallocate, please*/
}
