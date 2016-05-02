#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <stdlib.h>


// Thread block size
#define BLOCK_SIZE 16 //submatrix size 
#define N 16 //8192 matrix size is N*N 
__global__ void matMult(float* a, float* b, float *t, int n, float* c);
void PrintMatrix(float* a, int n);
void InitMatrix(float* a, int n);


int main (int argc, char* argv[])
{
	int numBytes = N * N * sizeof(float);
	
	//allocate host memory 
	float* a = new float[N*N];
	float* b = new float[N*N];
	float* c = new float[N*N];

	//init matricies
	InitMatrix(a, N);
	InitMatrix(b, N);

	//allocate device memory 
	float* adev = NULL;
	float* bdev = NULL;
	float* cdev = NULL;
	float* tdev = NULL;
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);
	cudaMalloc((void**)&tdev, numBytes);

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

	matMult<<<blocks, threads>>>(adev, bdev, tdev, N, cdev);

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
