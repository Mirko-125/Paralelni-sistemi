#include "cuda_runtime.h"

struct dim
{
    int x;
    int y;
};

dim threadIdx;
dim blockIdx;
dim blockDim;
dim gridDim;