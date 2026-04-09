#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0
#define N 8

int main(int argc, char *argv[]) // p = 4 | 2 | 1
{
    int i, j, k, id, p;
    double A[N][N], B[N][N], C[N][N];
    MPI_Datatype A_blocks, B_blocks, A_blocks_plus, B_blocks_plus;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == ROOT)
    {
        double val = 0.0;
        printf("Matrix A (%dx%d):\n", N, N);
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                A[i][j] = ++val;
                printf("%6.1f ", A[i][j]);
            }
            printf("\n");
        }
        val = 64.0;
        printf("\nMatrix B (%dx%d):\n", N, N);
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                B[i][j] = val--;
                printf("%6.1f ", B[i][j]);
            }
            printf("\n");
        }
    }

    double *Ablock = (double *)malloc(N * N / p * sizeof(double));
    double *Bblock = (double *)malloc(N * N / p * sizeof(double));

    MPI_Type_vector(N / p, N, N * p, MPI_DOUBLE, &A_blocks);
    MPI_Type_create_resized(A_blocks, 0, N * sizeof(double), &A_blocks_plus);
    MPI_Type_commit(&A_blocks_plus);

    MPI_Type_vector(N * N / p, 1, p, MPI_DOUBLE, &B_blocks);
    MPI_Type_create_resized(B_blocks, 0, sizeof(double), &B_blocks_plus);
    MPI_Type_commit(&B_blocks_plus);

    MPI_Scatter(A, 1, A_blocks_plus, Ablock, N * N / p, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(B, 1, B_blocks_plus, Bblock, N * N / p, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

#pragma region Visualization
    MPI_Barrier(MPI_COMM_WORLD);
    for (int rank = 0; rank < p; rank++)
    {
        if (id == rank)
        {
            printf("\n[P%d] block_A:\n", id);
            for (i = 0; i < N / p; i++)
            {
                printf("  row %d: ", i);
                for (j = 0; j < N; j++)
                    printf("%6.1f ", Ablock[i * N + j]);
                printf("\n");
            }
            printf("[P%d] block_B:\n", id);
            for (i = 0; i < N; i++)
            {
                printf("  row %d: ", i);
                for (j = 0; j < N / p; j++)
                    printf("%6.1f ", Bblock[i * (N / p) + j]);
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#pragma endregion Visualization

    double *mid_C = (double *)calloc(N / p * N / p, sizeof(double));
    int bblock_owner = id;

    for (i = 0; i < N / p; i++)
        for (j = 0; j < N / p; j++)
            for (k = 0; k < N; k++)
            {
                mid_C[i * N / p + j] += Ablock[i * N + k] * Bblock[k * N / p + j];
            }

#pragma region Step_Visualization
    MPI_Barrier(MPI_COMM_WORLD);
    for (int rank = 0; rank < p; rank++)
    {
        if (id == rank)
        {
            printf("\n[P%d] Step %d — using Bblock from P%d, mid_C (%dx%d):\n",
                   id, step, bblock_owner, N / p, N / p);
            for (i = 0; i < N / p; i++)
            {
                printf("  row %d: ", i);
                for (j = 0; j < N / p; j++)
                    printf("%8.1f ", mid_C[i * (N / p) + j]);
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#pragma endregion Step_Visualization

    MPI_Reduce(mid_C, C, N * N, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int rank = 0; rank < p; rank++)
    {
        if (id == rank)
        {
            printf("\n[P%d] mid_C (%dx%d):\n", id, N / p, N / p);
            for (i = 0; i < N / p; i++)
            {
                printf("  row %d: ", i);
                for (j = 0; j < N / p; j++)
                    printf("%8.1f ", mid_C[i * (N / p) + j]);
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (id == ROOT)
    {
        printf("\nMatrix C (%dx%d):\n", N, N);
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
                printf("%8.1f ", C[i][j]);
            printf("\n");
        }
    }

    MPI_Type_free(&A_blocks);
    MPI_Type_free(&A_blocks_plus);
    MPI_Type_free(&B_blocks);
    MPI_Type_free(&B_blocks_plus);
    return 0;
}
