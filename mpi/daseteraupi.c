#include <mpi.h>
#include <stdio.h>
#include <math.h>

#define ROOT 0
#define PI 3.1415926535897932384626433832795

int main(int argc, char **argv)
{
    int n, myid, numprocs, i;
    double mypi, pi, h, sum, x;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (myid == ROOT)
    {
        printf("Enter the number of intervals: ");
        fflush(stdout);
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    h = 1.0 / (double)n;
    sum = 0.0;

    for (i = myid + 1; i <= n; i += numprocs)
    {
        x = h * ((double)i - 0.5);
        sum += 4.0 / (1.0 + x * x);
    }

    mypi = h * sum;
    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (myid == ROOT)
    {
        printf("Approximation of pi is %.16f\n", pi);
    }

    MPI_Finalize();
    return 0;
}