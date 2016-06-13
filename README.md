atrix-matrix multiplication is a good example for parallel computation.

#####Software:
* Windows 7 32-bit
* Visual Studio 2013
* CUDA Toolkit 6.5 (<https://developer.nvidia.com/cuda-toolkit-65>)

#####Hardware:
* NVIDIA GeForce GT 430

#####Documantation (After installation of toolkit it will be available as well)
* `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\doc`

#####Samples
* `C:\ProgramData\NVIDIA Corporation\CUDA Samples\v6.5`

#####Content:
Calculate (A\*const1 - B\*const2) + (A\*B), where A, B square matricies 4096\*4096 with integer typed elements
1. using global memory
2. using shared memory
3. using CUBLAS library