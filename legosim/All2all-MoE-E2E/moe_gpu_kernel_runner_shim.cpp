#include <cuda_runtime_api.h>
#include <stddef.h>
static dim3 g_grid(1,1,1), g_block(1,1,1);
static size_t g_shared = 0;
static void *g_stream = 0;
extern "C" unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream) {
    g_grid = gridDim; g_block = blockDim; g_shared = sharedMem; g_stream = (void*)stream; return 0;
}
extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {
    if (gridDim) *gridDim = g_grid;
    if (blockDim) *blockDim = g_block;
    if (sharedMem) *sharedMem = g_shared;
    if (stream) *(void**)stream = g_stream;
    return cudaSuccess;
}
