#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER 0

#define K 4
#define M 5
#define L 6

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

    double *rowA = (double *)malloc(sizeof(double) * (K / p) * M);

    MPI_Datatype row_size_t, row_resize_t;

    MPI_Type_vector(K / p, M, M * p, MPI_DOUBLE, &row_size_t);
    MPI_Type_create_resized(row_size_t, 0, sizeof(double) * M, &row_resize_t);
    MPI_Type_commit(&row_resize_t);

    MPI_Scatter(&A[0][0], 1, row_resize_t, rowA, (K / p) * M, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Bcast(&B[0][0], M * L, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int rank = 0; rank < p; rank++)
    {
        if (myrank == rank)
        {
            printf("\n[P%d] rowA (%d rows x %d cols):\n", myrank, K / p, M);
            for (i = 0; i < K / p; i++)
            {
                printf("  row %d: ", i);
                for (j = 0; j < M; j++)
                    printf("%6.1f ", rowA[i * M + j]);
                printf("\n");
            }

            printf("[P%d] B (%d rows x %d cols):\n", myrank, M, L);
            for (i = 0; i < M; i++)
            {
                printf("  row %d: ", i);
                for (j = 0; j < L; j++)
                    printf("%6.1f ", B[i][j]);
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double *midC = (double *)malloc((K / p) * L * sizeof(double));

    for (i = 0; i < (K / p); i++)
        for (j = 0; j < L; j++)
            for (k = 0; k < M; k++)
                midC[i * L + j] += rowA[i * M + k] * B[k][j];

    printf("\n[P%d] midC (%d rows x %d cols):\n", myrank, K / p, L);
    for (i = 0; i < K / p; i++)
    {
        printf("  row %d: ", i);
        for (j = 0; j < L; j++)
            printf("%6.1f ", midC[i * L + j]);
        printf("\n");
    }
    /* WIP
    int block_lens[K / p];
    int displacements[K / p];

    for (i = 0; i < K / p; i++)
    {
        block_lens[i] = L;
        displacements[i] = (myrank + i * p) * L;
    }

    MPI_Datatype gather_t, gather_resized_t;


    MPI_Type_indexed(K / p, block_lens, displacements, MPI_DOUBLE, &gather_t);
    MPI_Type_create_resized(gather_t, 0, K * L * sizeof(double), &gather_resized_t);
    MPI_Type_commit(&gather_resized_t);

    MPI_Gather(midC, (K / p) * L, MPI_DOUBLE, &C, 1, gather_resized_t, MASTER, MPI_COMM_WORLD);

    MPI_Type_free(&gather_t);
    MPI_Type_free(&gather_resized_t);

    */

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

    free(rowA);
    free(midC);

    MPI_Type_free(&row_size_t);
    MPI_Type_free(&row_resize_t);
    MPI_Finalize();
    return 0;
}
