#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef GPGPUSIM_LEGACY_LAUNCH
extern "C" cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset);
extern "C" cudaError_t CUDARTAPI cudaLaunch(const void *func);
#endif

__global__ void expert_mlp_kernel(float *a, float *b, float *c, int tokens, int hidden, int inter, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * hidden;
    if (idx < total) {
        float acc = 0.0f;
        float aval = a[idx];
        for (int r = 0; r < repeat; ++r) {
            for (int j = 0; j < inter; ++j) acc = aval * b[j] + acc;
        }
        c[idx] = acc;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <compute_tokens> <hidden> <inter> [repeat]\n", argv[0]);
        return 2;
    }
    int compute_tokens = atoi(argv[1]);
    int hidden = atoi(argv[2]);
    int inter = atoi(argv[3]);
    int repeat = argc > 4 ? atoi(argv[4]) : 1;
    if (compute_tokens < 1) compute_tokens = 1;
    if (hidden < 1) hidden = 1;
    if (inter < 1) inter = 1;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    size_t a_count = (size_t)compute_tokens * hidden;
    size_t b_count = (size_t)hidden * inter;
    fprintf(stderr, "[GPGPU runner] alloc compute_tokens=%d hidden=%d inter=%d repeat=%d\n", compute_tokens, hidden, inter, repeat);
    cudaMalloc((void**)&a, a_count * sizeof(float));
    cudaMalloc((void**)&b, b_count * sizeof(float));
    cudaMalloc((void**)&c, a_count * sizeof(float));
    cudaMemset(a, 1, a_count * sizeof(float));
    cudaMemset(b, 1, b_count * sizeof(float));
    int threads = 128;
    int blocks = (int)((a_count + threads - 1) / threads);
    fprintf(stderr, "[GPGPU runner] kernel_launch blocks=%d threads=%d\n", blocks, threads);
#ifdef GPGPUSIM_LEGACY_LAUNCH
    // GPGPU-Sim's CUDA-11.5 runtime in this tree still expects the pre-cudaLaunchKernel
    // launch stack protocol.  The modern NVCC triple-chevron stub calls cudaLaunchKernel
    // directly and trips cudaSetupArgumentInternal("empty launch stack").
    cudaConfigureCall(dim3(blocks), dim3(threads), 0, 0);
    size_t off = 0;
#define PUSH_ARG(v) do { cudaSetupArgument(&(v), sizeof(v), off); off += sizeof(v); } while (0)
    PUSH_ARG(a);
    PUSH_ARG(b);
    PUSH_ARG(c);
    PUSH_ARG(compute_tokens);
    PUSH_ARG(hidden);
    PUSH_ARG(inter);
    PUSH_ARG(repeat);
#undef PUSH_ARG
    cudaLaunch((const void*)expert_mlp_kernel);
#else
    expert_mlp_kernel<<<blocks, threads>>>(a, b, c, compute_tokens, hidden, inter, repeat);
#endif
    fprintf(stderr, "[GPGPU runner] sync\n");
    cudaError_t e = cudaDeviceSynchronize();
    fprintf(stderr, "[GPGPU runner] done status=%d\n", (int)e);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return e == cudaSuccess ? 0 : 1;
}
