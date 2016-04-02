#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <stdlib.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct
{
	int width;
	int height;
	float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 32
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
void PrintMatrix(Matrix A);
void InitMatrix(Matrix* A);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) 
{
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(err));
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err));
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);
	MatMulKernel << <dimGrid, dimBlock >> >(d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));
	// Read C from device memory
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n", cudaGetErrorString(err));
	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	// cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) 
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > A.height || col > B.width) return;

	for (int e = 0; e < A.width; ++e) 
	{
		Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
	}
	C.elements[row * C.width + col] = Cvalue;
}

// Usage: multNoShare a1 a2 b2
int main(int argc, char* argv[])
{
	Matrix A, B, C;
	int a1, a2, b1, b2;
	int size = 1000;
	// Read some values from the commandline
	a1 = size; /* Height of A */
	a2 = size; /* Width of A */
	b1 = a2; /* Height of B = Width of A*/
	b2 = size; /* Width of B */
	A.height = a1;
	A.width = a2;
	A.elements = (float*)malloc(A.width * A.height * sizeof(float));
	B.height = b1;
	B.width = b2;
	B.elements = (float*)malloc(B.width * B.height * sizeof(float));
	C.height = A.height;
	C.width = B.width;
	C.elements = (float*)malloc(C.width * C.height * sizeof(float));

	InitMatrix(&A);
	InitMatrix(&B);
	MatMul(A, B, C);
	
	PrintMatrix(C);
	
	getch();
	free(&A); free(&B); free(&C);
}void PrintMatrix(Matrix A){	printf("%.0f ", A.elements[(A.height-1)*A.width + (A.width-1)]);	/*for (int i = 0; i < A.height; i++)
	{
		for (int j = 0; j < A.width; j++)
		{
			printf("%.0f ", A.elements[i*A.width + j]);
		}
		printf("\n");
	}*/	printf("\n");}void InitMatrix(Matrix* A){	for (int i = 0; i < A->height; i++)
	{
		for (int j = 0; j < A->width; j++)
		{
			A->elements[i*A->width + j] = 1;		}	}}