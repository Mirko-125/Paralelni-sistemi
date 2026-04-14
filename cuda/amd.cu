#include <cuda_runtime.h>
#include <stdio.h>

// AMD Pufla zadatak: CUDA Edition!!

constexpr int N = 1024;
constexpr int THREADS = 256;

__global__ void addVectors(int *a, int *b, int *c, int SIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid<N)
    {
        c[tid] = a[tid] + b[tid];
        tid+= blockIdx.x * blockDim.x;
    }
}

int main(void)
{
    int *h_A, *h_B, *h_C;

    h_A = new int[N];
    h_B = new int[N];
    h_C = new int[N];

    for (int i = 0; i < N; i++)
    {
        h_A[i]=i;
        h_B[i]=i*2;
    }
    
    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A,N*sizeof(int));
    cudaMalloc((void**)&d_B,N*sizeof(int));
    cudaMalloc((void**)&d_C,N*sizeof(int));

    cudaMemcpy(&d_A,&h_A,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(&d_B,&d_A,N*sizeof(int),cudaMemcpyHostToDevice);

    dim3 threads_by_block(THREADS);
    dim3 blocks_by_grid((N+THREADS-1)/THREADS); // ili threads_by_block.x

    addVectors<<<blocks_by_grid,threads_by_block>>>(d_A,d_B,d_C,N);

    cudaMemcpy(&h_C,&d_C,N*sizeof(int),cudaMemcpyDeviceToHost);

    if(h_A[25]+h_B[25]==h_C[25])
    {
        printf("It just works.");
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}