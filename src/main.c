#include <stdio.h>
#include <string.h>
#include "spmv.cuh"
#include "genresult.cuh"
#include "mmio.h"

void logError(const char * errArg, const char * eMsg){
	if(eMsg != NULL)
		printf("Error: %s\n", eMsg);
	if(errArg != NULL)
		printf("Error found near: '%s'\n", errArg);
	puts("USAGE: spmv -mat [matrix file] -ivec [vector file] -alg [atomic|segment|design] -blockSize [blocksize] -blockNum [blocknum]");
	puts("Where the order of the parameters and string case do not matter");
	puts("Though the blockSize is optional (defaults to 1024)");
	puts("And the algorithms are:");
	puts("     - atomic =  simple atomics based approach");
	puts("     - segment = simple segment based scan approach");
	puts("     - design = design implementation");
}

typedef enum{CMDLN_ARG_NULL, CMDLN_ARG_MAT = 1, CMDLN_ARG_VEC = 2, CMDLN_ARG_ALG = 4, CMDLN_ARG_BLOCK = 8, CMDLN_ARG_BLOCKNUM = 16, CMDLN_ARG_ERR = 32} CmdLnArg;

CmdLnArg getArgType(const char * argv){
	if(strcasecmp(argv, "-mat") == 0)
		return CMDLN_ARG_MAT;
	else if(strcasecmp(argv, "-ivec") == 0)
		return CMDLN_ARG_VEC;
	else if(strcasecmp(argv, "-alg") == 0)
		return CMDLN_ARG_ALG;
	else if(strcasecmp(argv, "-blockSize") == 0)
		return CMDLN_ARG_BLOCK;
	else if(strcasecmp(argv, "-blockNum") == 0)
		return CMDLN_ARG_BLOCKNUM;
	else
		return CMDLN_ARG_ERR;
}

typedef enum {ALG_ATOMIC, ALG_SEGMENT, ALG_DESIGN} AlgType;

int populateAlgType(const char * argv, AlgType * toPop){
	if(strcasecmp(argv, "atomic") == 0){
		*toPop = ALG_ATOMIC;
		return 1;
	}else if(strcasecmp(argv, "segment") == 0){
		*toPop = ALG_SEGMENT;
		return 1;
	}else if(strcasecmp(argv, "design") == 0){
		*toPop = ALG_DESIGN;
		return 1;
	}else return 0;
}

int cmpTuple(const void * a, const void * b){
	return (*(((MTuple *)a)->r) - *(((MTuple *)b)->r));
}

MatrixInfo * transferMat(MatrixInfo * mat){

	MTuple * tupleMat = (MTuple *) malloc(mat->nz*sizeof(MTuple));

	for (int i = 0 ; i < mat->nz ; i++){
		tupleMat[i].r = &(mat->rIndex[i]);
		tupleMat[i].c = &(mat->cIndex[i]);
		tupleMat[i].v = &(mat->val[i]);
	}

	qsort(tupleMat, mat->nz, sizeof(MTuple), cmpTuple);

	MatrixInfo * newMat = (MatrixInfo *) malloc(sizeof(MatrixInfo));
	newMat->rIndex = (int *) malloc(mat->nz*sizeof(int));
	newMat->cIndex = (int *) malloc(mat->nz*sizeof(int));
	newMat->val = (float *) malloc(mat->nz*sizeof(float));

	newMat->M = mat->M;
	newMat->N = mat->N;
	newMat->nz = mat->nz;
	
	for(int i = 0 ; i < mat->nz ; i++){
		newMat->rIndex[i] = *(tupleMat[i].r);
		newMat->cIndex[i] = *(tupleMat[i].c);
		newMat->val[i] = *(tupleMat[i].v);
	}
	
	free(tupleMat);
	
	return newMat;
}

void convert2CSR(MatrixInfo * mat){
	int * CSRIndex = (int *) malloc((mat->M+1)*sizeof(int));

	// takes values 0 to M
	int currRow = 0;
	// number of nonzero elements seen so far
	int count = 0;

	CSRIndex[0] = 0;

	int i = 0;
	while(i < mat->nz){
		int elemRow = mat->rIndex[i];
		
		if( currRow != elemRow ){
				CSRIndex[currRow+1] = count;
				currRow++;
		}else{
			count++;
			i++;
		}
						
	}

	if(count == mat->nz){
		printf("Conversion from COO to CSR is a success!\n");
		CSRIndex[currRow+1] = count;
	}

	free(mat->rIndex);
	mat->rIndex = CSRIndex;
}



int doSpmv(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, AlgType how, int blockSize, int blockNum){
	MatrixInfo * newMat;
	switch(how){
		case ALG_ATOMIC:
			getMulAtomic(mat, vec, res, blockSize, blockNum);
			return 1;
		case ALG_SEGMENT:
			printf("Changing sparse matrix format to COO row major order...\n");
			newMat = transferMat(mat);
			printf("Conversion complete.\nStarting calculation...\n");
			getMulScan(newMat, vec, res, blockSize, blockNum);
			freeMatrixInfo(newMat);
			return 1;
		case ALG_DESIGN:
			printf("Changing sparse matrix format to CSR...\n");
			newMat = transferMat(mat);
			convert2CSR(newMat);
			printf("Starting calculation...\n"); 
			getMulDesign(newMat, vec, res, blockSize, blockNum);
			freeMatrixInfo(newMat);
			return 1;
		default:
			return 0;
	}
}

int verify(const int nz, const int M, const int *rIndex, const int *cIndex, const float *val, const float *vec, const float *res) {

	float *correct = (float*)malloc(sizeof(float) * M);
	memset(correct, 0, sizeof(float) * M);
	for (int i = 0; i < nz; ++i) {
		correct[rIndex[i]] += val[i] * vec[cIndex[i]];
	}

	int o = 0;

	for (int i = 0; i < M; ++i) {
		float l = correct[i] > 0 ? correct[i] : -1*correct[i];
		float m = res[i] > 0 ? res[i] : -1*res[i];
		float k = l - m > 0 ? l - m : m - l;
		float rel = k / l;
		if (rel > .01) {
			o++;
			printf("Yours - %f, correct - %f, Relative error - %f\n", res[i], correct[i], rel);
			printf("Row index is: %d\n", rIndex[i]);
		}
	}

	return o;
}

int main(int argc, char ** argv){
	if(argc != 11){
		logError(NULL, NULL);
		return 1;
	}

	//This is so that the arguments can be presented in any order with the blocksize defaulting to 1024
	int cumArgs = CMDLN_ARG_NULL;
	CmdLnArg argOrder[5];
	int i;
	for(i = 1; i < argc; i += 2){
		CmdLnArg currArg = getArgType(argv[i]);
		if(currArg == CMDLN_ARG_ERR || currArg & cumArgs){
			logError(argv[i], "Invalid or duplicate argument.");
			return 1;
		}else{
			argOrder[i/2] = currArg; //May the truncation be ever in our favor.
			cumArgs |= currArg;
		}
	}

	if(! (31 & cumArgs)){
		logError(NULL, "Missing arguments!");
		return 1;
	}

	char * mFile, * vFile;
	AlgType algo; //Si, debe ser algo!
	int blockSize = 1024;
	int blockNum = 8;

	for(i = 0; i < (argc - 1)/2; i++){
		switch(argOrder[i]){
			case CMDLN_ARG_ALG:
				if(!populateAlgType(argv[i * 2 + 2], &algo)){
					logError(argv[i * 2 + 2], "Unsupported algorithm");
					return 1;
				}
				break;
			case CMDLN_ARG_MAT:
				mFile = argv[i * 2 + 2];
				break;
			case CMDLN_ARG_VEC:
				vFile = argv[i * 2 + 2];
				break;
			case CMDLN_ARG_BLOCK:
				if(sscanf(argv[i * 2 + 2], "%d", &blockSize) != 1 || blockSize <= 0){
					logError(argv[i * 2 + 2], "Block size must be a positive integer (greater than 0)");
					return 1;
				}
				break;
			case CMDLN_ARG_BLOCKNUM:
				if(sscanf(argv[i * 2 + 2], "%d", &blockNum) != 1 || blockNum <= 0){
					logError(argv[i * 2 + 2], "Block num must be a positive integer (greater than 0)");
					return 1;
				}
				break;

			default:
				puts("Logic is literally broken. This should never be seen!");
		}
	}

	printf("Reading matrix from %s\n", mFile);
	MatrixInfo * matrix = read_file(mFile);
	if(matrix == NULL){
		logError(mFile, "Error regarding matrix file.");
		return 1;
	}

	printf("Reading vector from %s\n", vFile);
	MatrixInfo * vector = read_vector_file(vFile, matrix->N);
	if(vector == NULL){
		logError(mFile, "Error regarding vector file.");
		return 1;
	}

	MatrixInfo * product = initMatrixResult(matrix->M, blockSize);
	cudaError_t err;
	if(!doSpmv(matrix, vector, product, algo, blockSize, blockNum)
			|| (err = cudaDeviceSynchronize()) != cudaSuccess
			|| !writeVect(product, "output.txt")){

		printf("\x1b[31m%s\x1b[0m\n", cudaGetErrorString(err));
		logError(NULL, "Failed to produce output");
	} else {
		printf("Verifying...\n");
		int o = verify(matrix->nz, matrix->M, matrix->rIndex, matrix->cIndex, matrix->val, vector->val, product->val);
		printf("%d Error rows \n", o);
	}

	freeMatrixInfo(matrix);
	freeMatrixInfo(vector);
	freeMatrixInfo(product);

	puts("So long and thank you for the fish!");
	return 0;
}
