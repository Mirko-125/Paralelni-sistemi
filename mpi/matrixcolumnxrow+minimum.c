#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define ROOT 0

#define K 5
#define M 8
#define L 9

int main(int argc, char *argv[]) // p = 8 | 4 | 2 | 1
{
    int i, j, k, id, p;
    double A[K][M], B[M][L], C[K][L];
    MPI_Datatype column_type, row_type, column_resized, row_resized;
    struct
    {
        double value;
        int id;
    } local_minimum_A, global_minimumA;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == ROOT)
    {
        double val = 0.0;
        printf("Matrix A (%dx%d):\n", K, M);
        for (i = 0; i < K; i++)
        {
            for (j = 0; j < M; j++)
            {
                A[i][j] = ++val;
                printf("%6.1f ", A[i][j]);
            }
            printf("\n");
        }
        val = 72.0;
        printf("\nMatrix B (%dx%d):\n", M, L);
        for (i = 0; i < M; i++)
        {
            for (j = 0; j < L; j++)
            {
                B[i][j] = val--;
                printf("%6.1f ", B[i][j]);
            }
            printf("\n");
        }
    }

    double *block_A = (double *)malloc(K * M / p * sizeof(double));
    double *block_B = (double *)malloc(M * L / p * sizeof(double));
    double *mid_C = (double *)calloc(K * L, sizeof(double));

    MPI_Type_vector(K * M / p, 1, p, MPI_DOUBLE, &column_type);
    MPI_Type_create_resized(column_type, 0, sizeof(double), &column_resized);
    MPI_Type_commit(&column_resized);

    MPI_Type_vector(M / p, L, L * p, MPI_DOUBLE, &row_type);
    MPI_Type_create_resized(row_type, 0, L * sizeof(double), &row_resized);
    MPI_Type_commit(&row_resized);

    MPI_Scatter(A, 1, column_resized, block_A, K * M / p, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(B, 1, row_resized, block_B, M * L / p, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    local_minimum_A.value = block_A[0];
    local_minimum_A.id = id;

    for (i = 0; i < K; i++)
        for (j = 0; j < L; j++)
            for (k = 0; k < M / p; k++)
            {
                mid_C[i * L + j] += block_A[i * (M / p) + k] * block_B[k * L + j];
                if (block_A[i * (M / p) + k] < local_minimum_A.value)
                {
                    local_minimum_A.value = block_A[i * (M / p) + k];
                }
            }

    MPI_Reduce(mid_C, C, K * L, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    MPI_Reduce(&local_minimum_A, &global_minimumA, 1, MPI_DOUBLE_INT, MPI_MINLOC, ROOT, MPI_COMM_WORLD);

    MPI_Bcast(C, K * L, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&global_minimumA, 1, MPI_DOUBLE_INT, ROOT, MPI_COMM_WORLD);

    if (id == global_minimumA.id)
    {
        printf("\nMatrix C (%dx%d):\n", K, L);
        for (i = 0; i < K; i++)
        {
            for (j = 0; j < L; j++)
                printf("%8.1f ", C[i][j]);
            printf("\n");
        }
        printf("\nWith a process P%d having a minimal value of %6f in matrix A\n", global_minimumA.id, global_minimumA.value);
    }

    free(block_A);
    free(block_B);
    free(mid_C);
    MPI_Type_free(&column_type);
    MPI_Type_free(&column_resized);
    MPI_Type_free(&row_type);
    MPI_Type_free(&row_resized);
    MPI_Finalize();
    return 0;
}