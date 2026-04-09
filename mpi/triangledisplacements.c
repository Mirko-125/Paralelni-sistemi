#include <stdio.h>
#include <mpi.h>

#define N 10
#define ROOT 0

int main(int argc, char *argv[])
{
    int i, j;
    MPI_Status status;

    int p, myrank;

    double A[N][N];
    double T[N][N];

    int displacements[N];
    int block_lengths[N];

    MPI_Datatype index_mpi_t;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    for (i = 0; i < N; i++)
    {
        block_lengths[i] = N - i;
        displacements[i] = (N + 1) * i;
    }

    MPI_Type_indexed(N, block_lengths, displacements, MPI_DOUBLE, &index_mpi_t);
    MPI_Type_commit(&index_mpi_t);

    if (ROOT == myrank)
    {
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
            {
                A[i][j] = (double)(i + j);
            }

        printf("[P0] Matrix A before send:\n");
        for (i = 0; i < N; i++)
        {
            printf("  row %d: ", i);
            for (j = 0; j < N; j++)
                printf("%5.1f ", A[i][j]);
            printf("\n");
        }

        printf("\n[P0] Upper triangle only (what gets sent):\n");
        for (i = 0; i < N; i++)
        {
            printf("  row %d: ", i);
            for (j = 0; j < N; j++)
                printf("%5.1f ", j >= i ? A[i][j] : 0.0);
            printf("\n");
        }
        MPI_Send(A, 1, index_mpi_t, 1, 0, MPI_COMM_WORLD);
    }
    else if (myrank == 1)
    {
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                T[i][j] = 0.0;
        MPI_Recv(T, 1, index_mpi_t, ROOT, 0, MPI_COMM_WORLD, &status);

        printf("\n[P1] Matrix T after receive:\n");
        for (i = 0; i < N; i++)
        {
            printf("  row %d: ", i);
            for (j = 0; j < N; j++)
                printf("%5.1f ", T[i][j]);
            printf("\n");
        }
    }

    MPI_Type_free(&index_mpi_t);
    MPI_Finalize();
    return 0;
}