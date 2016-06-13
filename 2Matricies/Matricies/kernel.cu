#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>

// Thread block size
#define BLOCK_SIZE 64 //submatrix size 
#define N 4096 //matrix size is N*N 
#define MULTIPLIER_A 5
#define MULTIPLIER_B 8
__global__ void matMatMult(int* a, int* b, int n, int* c);
__global__ void matMultByConst(int* a, int k, int n, int* c);
__global__ void sumMat(int* a, int* b, int n, int* c); 
void PrintMatrix(int* a, int n);
void InitMatrix(int* a, int n, int initVal);

int main (int argc, char* argv[])
{
	int numBytes = N * N * sizeof(int);
	
	//allocate host memory 
	int* a = new int[N*N];
	int* b = new int[N*N];
	int* c = new int[N*N];

	//init matricies
	InitMatrix(a, N, 1);
	InitMatrix(b, N, 1);

	//allocate device memory 
	int* adev = NULL;
	int* bdev = NULL;
	int* cdev = NULL;
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);

	//set kernel launch configuration 
	dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks (N / threads.x, N / threads.y);

	//create cuda event handles 
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	//asynchronously issue work to the GPU (all to stream 0) 
	cudaEventRecord(start, 0);
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

	matMatMult << <blocks, threads >> >(adev, bdev, N, cdev); cudaDeviceSynchronize();
	matMultByConst << <blocks, threads >> >(adev, MULTIPLIER_A, N, adev); cudaDeviceSynchronize();
	matMultByConst << <blocks, threads >> >(bdev, MULTIPLIER_B, N, bdev); cudaDeviceSynchronize();
	sumMat << <blocks, threads >> >(adev, bdev, N, cdev); cudaDeviceSynchronize();

	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
	PrintMatrix(c, N);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	
	//print the cpu and gpu times 
	printf("time spent executing by the GPU: %.2f ms\n", gpuTime);

	//release resources 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	delete a;
	delete b;
	delete c;
	getch();
	return 0;
}

__global__ void matMatMult(int* a, int* b, int n, int* c)
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
	int sum = 0.0f;
	int ia = n * BLOCK_SIZE * by + n * ty; //a [i][0] 
	int ib = BLOCK_SIZE * bx + tx;

	//C = A*B 
	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
	{
		__shared__ int as[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ int bs[BLOCK_SIZE][BLOCK_SIZE];
		as[ty][tx] = a [ia + N * ty + tx];
		bs[ty][tx] = b [ib + N * ty + tx];
		__syncthreads (); // Убедимся, что подматрицы полностью загружены
		for ( int k = 0; k < BLOCK_SIZE; k++ )
		{
			sum += as [ty][k] * bs [k][tx];
		}
		__syncthreads (); // Убедимся, что подматрицы не используются
	}
	c [N * BLOCK_SIZE * by + BLOCK_SIZE * bx + N * ty + tx] = sum;
	//Write the block sub‐matrix to global memory; each thread writes one element 
}

__global__ void matMultByConst(int* a, int k, int n, int* c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		c[i*n + j] = a[i*n + j] * k;
	}
}

__global__ void sumMat(int* a, int* b, int n, int* c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		c[i*n + j] = a[i*n + j] + b[i*n + j] + c[i*n + j];
	}
}

/* UTILS */
void InitMatrix(int* a, int n, int initVal)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int k = n * i + j;
			a[k] = initVal;
		}
	}
}

void PrintMatrix(int* a, int n)
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
