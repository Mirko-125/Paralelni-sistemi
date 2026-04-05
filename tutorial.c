#include <stdio.h>
#include <mpi.h>
int main(int argc, char **argv)
{
    int myrank;
    MPI_Status status;
    int x, y;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0)
    {
        x = 3;
        MPI_Send(&x, 1, MPI_INT, 1, 47, MPI_COMM_WORLD);
        MPI_Recv(&y, 1, MPI_INT, 1, 69, MPI_COMM_WORLD, &status);
    }
    else if (myrank == 1)
    {
        x = 5;
        MPI_Recv(&y, 1, MPI_INT, 0, 47, MPI_COMM_WORLD, &status);
        MPI_Send(&x, 1, MPI_INT, 0, 69, MPI_COMM_WORLD);
    }
    printf("Proc %d y = %d", myrank, y);
    MPI_Finalize();
    return 0;
}