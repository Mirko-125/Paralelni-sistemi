#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define N 6 // N == p

int main(int argc, char *argv[])
{
    int rank, size, i, j;
    float A[N][N], B[N], C[N], local_c;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (MASTER == rank)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                A[i][j] = (float)(i + j);
            }
            B[i] = (float)(i + 1);
        }
    }

    float *local_a = (float *)malloc(N * sizeof(float));

    MPI_Scatter(A, N, MPI_FLOAT, local_a, N, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(B, N, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    local_c = 0.0;
    for (i = 0; i < N; i++)
    {
        local_c += local_a[i] * B[i];
    }

    MPI_Gather(&local_c, 1, MPI_FLOAT, C, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER)
    {
        printf("Result of matrix-vector multiplication:\n");
        for (i = 0; i < N; i++)
        {
            printf("%f\n", C[i]);
        }
    }
    free(local_a);
    MPI_Finalize();
    return 0;
}