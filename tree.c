#include <mpi.h>
#include <stdio.h>
#include <math.h>

#define ROOT 0

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Status status;
    int sum, val;

    int i, level, nextto, nlevels;

    int src, dst;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sum = rank;

    nlevels = (int)(log((double)size) / log(2.0));

    for (i = 0; i < nlevels; i++)
    {
        level = (int)pow(2.0, (double)i);
        if (rank % level == 0)
        {
            nextto = (int)pow(2.0, (double)i + 1);
            if (rank % nextto == 0)
            {
                src = rank + level;
                MPI_Recv(&val, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
                sum += val;
            }
            else
            {
                dst = rank - level;
                MPI_Send(&sum, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
            }
        }
    }
    if (ROOT == rank)
    {
        printf("Total sum is %d\n", sum);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}