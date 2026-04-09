## MPI

requirements:

[MPI Library](https://www.microsoft.com/en-us/download/details.aspx?id=105289)

build:

```bash
mpic++ -std=c++17 -O2 -o <dst>.exe <src>.c
```

run:

```bash
mpiexec -n <num_procs> ./<build>.exe
```

## CUDA

requirements:

[NVIDIA Graphics card](https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5090/)

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

build:

```bash
nvcc <src>.cu -o <dst>.exe
```

run:

```bash
<dst>.exe
```
