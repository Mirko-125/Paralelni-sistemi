#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int myrank, x, y;
    MPI_Request req;
    MPI_Status status;
    MPI_Init(&argc, &argv);                 /* Initialize MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); /* Get rank */
    if (myrank == 0)
    {
        x = 3;
        MPI_Irecv(&y, 1, MPI_INT, 1, 19, MPI_COMM_WORLD, &req);
        MPI_Send(&x, 1, MPI_INT, 1, 17, MPI_COMM_WORLD);
        MPI_Wait(&req, &status);
    }
    else if (myrank == 1)
    {
        x = 5;
        MPI_Irecv(&y, 1, MPI_INT, 0, 17, MPI_COMM_WORLD, &req);
        MPI_Send(&x, 1, MPI_INT, 0, 19, MPI_COMM_WORLD);
        MPI_Wait(&req, &status);
    }
    printf("Proc %d y = %d", myrank, y);
    MPI_Finalize();
    return 0;
}