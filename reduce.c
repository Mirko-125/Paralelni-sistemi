#include <stdio.h>
#include <mpi.h>
int main(int argc, char *argv[])
{
    int rank;
    int source, result, root;
    result = -1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    root = 0;
    source = rank + 1;
    // MPI_Reduce(&source, &result, 1, MPI_INT, MPI_PROD, root, MPI_COMM_WORLD);
    MPI_Scan(&source, &result, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
    printf("PE:%d MPI_PROD result is %d\n", rank, result);
    MPI_Finalize();
    return 0;
}