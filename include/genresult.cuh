#ifndef __GENRESULT_INCL
#define __GENRESULT_INCL
#include "mmio.h"

/* Gives the corresponding grid size for a provided block size.
 * Kernels can be configured with <<<getGridSize(blockSize, number_of_things), blockSize)>>>.*/
int getGridSize(int blockSize, int n);

/* Initializes a MatrixInfo to store the result vector.*/
MatrixInfo * initMatrixResult(int N, int blockSize);
#endif
