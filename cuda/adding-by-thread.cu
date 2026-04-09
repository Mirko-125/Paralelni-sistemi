#include <cuda_runtime.h>
#include <stdio.h>

constexpr int N = 256;
constexpr int SIZE = N * sizeof(int);

__global__ void add_by_block(int *a, int *b, int *c, int max_size)
{
    int tid = threadIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void)
{
    int host_first_vector[N], host_second_vector[N], host_output_vector[N];

    int *device_vector_left, *device_vector_right, *device_vector_output;

    for (int i = 0; i < N; i++)
    {
        host_first_vector[i] = i;
        host_second_vector[i] = 2 * i;
    }

    cudaMalloc((void **)&device_vector_left, SIZE);
    cudaMalloc((void **)&device_vector_right, SIZE);
    cudaMalloc((void **)&device_vector_output, SIZE);

    cudaMemcpy(device_vector_left, host_first_vector, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_right, host_second_vector, SIZE, cudaMemcpyHostToDevice);

    dim3 blocks_per_grid(1);
    dim3 threads_per_block(N);

    // GRID => limitirano na 1 blok sa 256 threada
    add_by_block<<<blocks_per_grid, threads_per_block>>>(device_vector_left, device_vector_right, device_vector_output, N);

    cudaMemcpy(host_output_vector, device_vector_output, SIZE, cudaMemcpyDeviceToHost);

    cudaFree(device_vector_left);
    cudaFree(device_vector_right);
    cudaFree(device_vector_output);

    for (int i = 0; i < N; i++)
    {
        if (host_first_vector[i] + host_second_vector[i] != host_output_vector[i])
        {
            printf("ALO doshlo je do greshke!!\n");
        }
        else
        {
            printf("Shljaka!!\n");
        }
    }

    return 0;
}