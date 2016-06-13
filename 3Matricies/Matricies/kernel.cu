#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <cublas_v2.h>

// Thread block size
#define BLOCK_SIZE 16 //submatrix size 
#define N 16 //8192 matrix size is N*N 
#define MULTIPLIER_A 5
#define MULTIPLIER_B 8
__global__ void matMultByConst(float* a, int k, int n, float* c);
__global__ void sumMat(float* a, float* b, int n, float* c);//sum A, B, C, result to C

void PrintMatrix(float* a, int n);
void InitMatrix(float* a, int n);
void gpu_blas_mmul(float *A, float *B, float *C, int n);

int main (int argc, char* argv[])
{
	// Allocate 3 arrays on CPU
	float *h_A = (float *)malloc(N * N * sizeof(float));
	float *h_B = (float *)malloc(N * N * sizeof(float));
	float *h_C = (float *)malloc(N * N * sizeof(float));

	// Allocate 3 arrays on GPU
	float* d_A = NULL;
	float* d_B = NULL;
	float* d_C = NULL;
	cudaMalloc((void**)&d_A, N * N * sizeof(float));
	cudaMalloc((void**)&d_B, N * N * sizeof(float));
	cudaMalloc((void**)&d_C, N * N * sizeof(float));
	
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

	cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
	gpu_blas_mmul(d_A, d_B, d_C, N);
	
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);
	matMultByConst << <blocks, threads >> >(d_A, MULTIPLIER_A, N, d_A); cudaDeviceSynchronize();
	matMultByConst << <blocks, threads >> >(d_B, MULTIPLIER_B, N, d_B); cudaDeviceSynchronize();
	sumMat << <blocks, threads >> >(d_A, d_B, N, d_C); cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
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

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(float *A, float *B, float *C, int n) {
     int lda=n,ldb=n,ldc=n;//m,n,k
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;
 
     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);
 
     // Do the actual multiplication
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, A, lda, B, ldb, beta, C, ldc);
 
     // Destroy the handle
     cublasDestroy(handle);
}

__global__ void matMultByConst(float* a, int k, int n, float* c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		c[i*n + j] = a[i*n + j] * k;
	}
}

__global__ void sumMat(float* a, float* b, int n, float* c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		c[i*n + j] = a[i*n + j] + b[i*n + j] + c[i*n + j];
	}
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
			printf("[%d,%d]: %d\n", i, j, a[k]);
		}
	}
	printf("\n");
}
