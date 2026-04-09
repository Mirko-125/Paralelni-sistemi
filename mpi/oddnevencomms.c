#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int i, rank, size, color, key, newrank, newsize;

    MPI_Comm newcomm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    color = rank % 2;
    key = 7;

    MPI_Comm_split(MPI_COMM_WORLD, color, 7, &newcomm);
    MPI_Comm_size(newcomm, &newsize);
    MPI_Comm_rank(newcomm, &newrank);

    printf("[Rank %d] => %s | new_rank: %d | new_size: %d\n",
           rank,
           color == 0 ? "EVEN" : "ODD ",
           newrank,
           newsize);

    MPI_Finalize();
    return 0;
}