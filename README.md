## MPI

build:

```bash
mpic++ -std=c++17 -O2 -o <dst>.exe <src>.c
```

run:

```bash
mpiexec -n <num_procs> ./<build>.exe
```
