#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int i, p, rank, even_n, even_id, odd_n, odd_id;
    MPI_Group world_group, odd_group, even_group;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int members[p];

    even_n = (p + 1) / 2;
    odd_n = p - even_n;

    for (i = 0; i < even_n; i++)
    {
        members[i] = 2 * i;
    }

    MPI_Group_incl(world_group, even_n, members, &even_group);
    MPI_Group_rank(even_group, &even_id);

    MPI_Group_excl(world_group, even_n, members, &odd_group);
    MPI_Group_rank(odd_group, &odd_id);

    if (even_id == MPI_UNDEFINED)
        printf("Rank %d => ODD, odd_rank: %d\n", rank, odd_id);
    else
        printf("Rank %d => EVEN, even_rank: %d\n", rank, even_id);

    MPI_Finalize();
    return 0;
}