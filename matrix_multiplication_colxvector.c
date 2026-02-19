#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define M 8
#define K 6

int main(int argc, char **argv)
{
    int id, p;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    double A[M][K];
    double B[K];

    int localRows = (M + p - 1 - id) / p;

    double *localA = (double *)malloc(localRows * K * sizeof(double));
    double *localC = (double *)malloc(localRows * sizeof(double));
    double *localColumnProd = (double *)malloc(K * sizeof(double));

    double C[M];
    double columnProd[K];

    if (id == MASTER)
    {
        double val = 1.0;
        printf("Matrix A:\n");
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < K; j++)
            {
                A[i][j] = val++;
                printf("%6.1f ", A[i][j]);
            }
            printf("\n");
        }

        printf("\nVector B:\n");
        for (int i = 0; i < K; i++)
        {
            B[i] = (double)(i + 1);
            printf("%6.1f ", B[i]);
        }
        printf("\n\n");
    }

    MPI_Bcast(B, K, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Datatype rows_t, resized_rows_t;

    MPI_Type_vector(localRows, K, p * K, MPI_DOUBLE, &rows_t);
    MPI_Type_create_resized(rows_t, 0, K * sizeof(double), &resized_rows_t);
    MPI_Type_commit(&resized_rows_t);

    MPI_Scatter(
        A,
        1,
        resized_rows_t,
        localA,
        localRows * K,
        MPI_DOUBLE,
        MASTER,
        MPI_COMM_WORLD);

    for (int i = 0; i < localRows; i++)
        localC[i] = 0.0;

    for (int j = 0; j < K; j++)
        localColumnProd[j] = 1.0;

    for (int i = 0; i < localRows; i++)
    {
        for (int j = 0; j < K; j++)
        {
            localC[i] += localA[i * K + j] * B[j];
            localColumnProd[j] *= localA[i * K + j];
        }
    }

    double *partialC = (double *)malloc(M * sizeof(double));
    for (int i = 0; i < M; i++)
        partialC[i] = 0.0;

    for (int i = 0; i < localRows; i++)
    {
        partialC[id + i * p] = localC[i];
    }

    MPI_Reduce(
        partialC,
        C,
        M,
        MPI_DOUBLE,
        MPI_SUM,
        MASTER,
        MPI_COMM_WORLD);

    MPI_Reduce(
        localColumnProd,
        columnProd,
        K,
        MPI_DOUBLE,
        MPI_PROD,
        MASTER,
        MPI_COMM_WORLD);

    if (id == MASTER)
    {
        printf("Rezultat c = A * b:\n");
        for (int i = 0; i < M; i++)
            printf("%8.2f ", C[i]);
        printf("\n\n");

        printf("Proizvod elemenata po kolonama matrice A:\n");
        for (int j = 0; j < K; j++)
            printf("%12.2f ", columnProd[j]);
        printf("\n");
    }

    MPI_Type_free(&rows_t);
    MPI_Type_free(&resized_rows_t);
    free(localA);
    free(localC);
    free(localColumnProd);
    free(partialC);

    MPI_Finalize();
    return 0;
}
