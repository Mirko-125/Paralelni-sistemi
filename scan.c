#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int source, result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("PE:%d SUM %d \n", rank, rank + 1);
    }

    source = rank + 1;
    MPI_Scan(&source, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("PE:%d SUM %d \n", rank, result);
    MPI_Finalize();
    return 0;
}