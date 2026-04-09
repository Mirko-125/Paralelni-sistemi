#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER 0

#define K 4
#define M 8
#define L 10

int main(int argc, char *argv[])
{
    int i, j, k, p, myrank;

    double A[K][M];
    double B[M][L];
    double C[K][L];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (MASTER == myrank)
    {
        double val = 0.0;
        for (i = 0; i < K; i++)
            for (j = 0; j < M; j++)
                A[i][j] = ++val;
        val = 0.0;
        for (i = 0; i < M; i++)
            for (j = 0; j < L; j++)
                B[i][j] = ++val;

        printf("[P0] Matrix A (%dx%d):\n", K, M);
        for (i = 0; i < K; i++)
        {
            printf("  row %d: ", i);
            for (j = 0; j < M; j++)
                printf("%6.1f ", A[i][j]);
            printf("\n");
        }

        printf("\n[P0] Matrix B (%dx%d):\n", M, L);
        for (i = 0; i < M; i++)
        {
            printf("  row %d: ", i);
            for (j = 0; j < L; j++)
                printf("%6.1f ", B[i][j]);
            printf("\n");
        }
    }

    double *columns_o_B = (double *)malloc(sizeof(double) * (M / p) * L);

    MPI_Datatype column_size_t, column_resize_t;

    MPI_Type_vector((M * L) / p, 1, p, MPI_DOUBLE, &column_size_t);
    MPI_Type_create_resized(column_size_t, 0, sizeof(double), &column_resize_t);
    MPI_Type_commit(&column_resize_t);

    MPI_Scatter(B, 1, column_resize_t, columns_o_B, (M / p) * L, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Bcast(A, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    double *midC = (double *)malloc((K / p) * L * sizeof(double));

    for (i = 0; i < (K / p); i++)
        for (j = 0; j < L; j++)
            for (k = 0; k < M; k++)
                midC[i * L + j] += A[i][k] * columns_o_B[k * L + j];

    printf("\n[P%d] midC (%d rows x %d cols):\n", myrank, K / p, L);
    for (i = 0; i < K / p; i++)
    {
        printf("  row %d: ", i);
        for (j = 0; j < L; j++)
            printf("%6.1f ", midC[i * L + j]);
        printf("\n");
    }

    MPI_Gather(midC, K / p * L, MPI_DOUBLE, C, K / p * L, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    if (MASTER == myrank)
    {
        printf("\n[P%d] C (%d rows x %d cols):\n", myrank, K, L);
        for (i = 0; i < K; i++)
        {
            printf("  row %d: ", i);
            for (j = 0; j < L; j++)
                printf("%6.1f ", C[i][j]);
            printf("\n");
        }
    }
    MPI_Finalize();
    return 0;
}