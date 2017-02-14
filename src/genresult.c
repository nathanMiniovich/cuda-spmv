#include <stdlib.h>
#include <math.h>
#include "genresult.cuh"

int getGridSize(int blockSize, int n){
    return (int) ceil((float)n/blockSize);
}

MatrixInfo * initMatrixResult(int N, int blockSize){
    MatrixInfo * temp = (MatrixInfo *) malloc(sizeof(MatrixInfo));
    temp->val = (float *) malloc(N*sizeof(float));
    temp->N = 1;
    temp->M = N;
    temp->nz = N;
    temp->rIndex = NULL;
    temp->cIndex = NULL;
    return temp;
}
