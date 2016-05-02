#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <cublas_v2.h>

// Thread block size
#define BLOCK_SIZE 16 //submatrix size 
#define N 16 //8192 matrix size is N*N 


void PrintMatrix(float* a, int n);
void InitMatrix(float* a, int n);
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n)

int main (int argc, char* argv[])
{
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
	
	float *h_A = (float *)malloc(N * N * sizeof(float));
	float *h_B = (float *)malloc(N * N * sizeof(float));
	float *h_C = (float *)malloc(N * N * sizeof(float));

	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, N * N * sizeof(float));
	cudaMalloc(&d_B, N * N * sizeof(float));
	cudaMalloc(&d_C, N * N * sizeof(float));
	
	// Fill the arrays A and B on GPU with random numbers

	//init matricies
	InitMatrix(h_A, N);
	InitMatrix(h_B, N);
	cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

	//create cuda event handles 
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	//asynchronously issue work to the GPU (all to stream 0) 
	cudaEventRecord(start, 0);

	//C = A*B
	gpu_blas_mmul(d_A, d_B, d_C, N, N, N);
	cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	//C = C*C
	cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice);
	gpu_blas_mmul(d_C, d_C, d_C, N, N, N);
	cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	
	//A = A*2, B = B*3
	float *mymatrix, *d_mymatrix;
	int size = M*N*sizeof(float);
	mymatrix = (float *)malloc(size);
	cudaMalloc((void **)&d_mymatrix, size);
	... (cublas/handle setup)
	cublasSetVector(N*N, sizeof(float), mymatrix, 1, d_mymatrix, 1);
	float alpha = 5.0;
	cublasSscal(handle, M*N, &alpha, d_mymatrix, 1); 
	
	//C = A - B + C

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	//print the cpu and gpu times 
	printf("time spent executing by the GPU: %.2f ms\n", gpuTime);
	// Copy (and print) the result on host memory
	
	PrintMatrix(h_C, N);

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);  
	//release resources 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);
	getch();
	return 0;
}

__global__ void matMult(float* a, float* b, float* t, int n, float* c)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = N * BLOCK_SIZE * by;
	int aEnd = aBegin + N - 1;
	int bBegin = BLOCK_SIZE * bx;
	int aStep = BLOCK_SIZE;
	int bStep = BLOCK_SIZE * N;
	float sum = 0.0f;
	int ia = n * BLOCK_SIZE * by + n * ty; //a [i][0] 
	int ib = BLOCK_SIZE * bx + tx;

	//C = A*B 
	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
	{
		__shared__ float as [BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float bs [BLOCK_SIZE][BLOCK_SIZE];
		as[ty][tx] = a [ia + N * ty + tx];
		bs[ty][tx] = b [ib + N * ty + tx];
		__syncthreads (); // Убедимся, что подматрицы полностью загружены
		for ( int k = 0; k < BLOCK_SIZE; k++ )
		{
			sum += as [ty][k] * bs [k][tx];
		}
		__syncthreads (); // Убедимся, что подматрицы никому больше не нужны
	}
	c [N * BLOCK_SIZE * by + BLOCK_SIZE * bx + N * ty + tx] = sum;
	//Write the block sub‐matrix to global memory; each thread writes one element 
	c[ic + n * ty + tx] = sum;
	__syncthreads();

	//C = C*C 
	sum = 0.0f;
	int ic1 = ia, ic2 = ib;
	for (int k = 0; k < n; k++)
	{
		sum += c[ic1 + k] * c[ic2 + k*n];
	}
	
	ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	c[ic + n * ty + tx] = sum;
	__syncthreads();

	//A = A*2, B = B*3 
	for (int k = 0; k < n; k++)
	{
		a[ia + k] *= 2;
		b[ib + k*n] *= 3;
	}
	__syncthreads();
	
	//C = A - B + C
	sum = 0.0f;
	ic = ia;
	for (int k = 0; k < n; k++)
	{
		sum = a[ia + k] - b[ib + k*n] + c[ic + k];
	}

	// Write the block sub‐matrix to global memory; each thread writes one element 
	ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	c[ic + n * ty + tx] = sum;
	__syncthreads();
}


// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;
 
     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);
 
     // Do the actual multiplication
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 
     // Destroy the handle
     cublasDestroy(handle);
}



/* UTILS */

void InitMatrix(float* a, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int k = n * i + j;
			a[k] = 1;
		}
	}
}

void PrintMatrix(float* a, int n)
{
	int rowStep = (n - 1) / 2;
	int colStep = (n - 1) / 2;

	for (int i = 0; i < n; i += rowStep)
	{
		for (int j = 0; j < n; j += colStep)
		{
			int k = n * i + j;
			printf("[%d,%d]: %.1f\n", i, j, a[k]);
		}
	}
	printf("\n");
}
