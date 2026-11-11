#include "apis_cu.h"
#include "sync_protocol.h"
#include "pipe_comm.h"

#include <cstring>
#include <vector>

static InterChiplet::PipeComm g_pipe_comm;
static InterChiplet::TimeType g_cycle = 0;

// The external GPGPU-Sim runner executes in a child process, so its simulated
// cycles do not automatically update this worker's InterChiplet timestamp.
// Fold the measured kernel cycles into the timestamp used by the next
// writeSync/readSync.  Do not call cycleSync() here: Interchiplet's CYCLE
// handler is intentionally one-way and would not send the SYNC reply that
// cycleSync() waits for.
extern "C" void advanceInterchipletCycles(unsigned long long cycles) {
    g_cycle += static_cast<InterChiplet::TimeType>(cycles);
}

// CUDA 12 host stubs use __cudaPushCallConfiguration/__cudaPopCallConfiguration
// around cudaLaunchKernel().  The bundled GPGPU-Sim libcudart has incomplete
// implementations for this ABI: Push has no return value and Pop does not hand
// the launch dimensions back to the CUDA-generated host stub.  That can make a
// real GPGPU-Sim worker spin at launch time instead of executing the kernel.
// Keep a tiny app-local implementation so CUDA 12-generated host code can still
// drive GPGPU-Sim's cudaLaunchKernel path correctly.
static dim3 g_last_grid_dim(1, 1, 1);
static dim3 g_last_block_dim(1, 1, 1);
static size_t g_last_shared_mem = 0;
static void *g_last_stream = nullptr;

extern "C" unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                           size_t sharedMem,
                                                           struct CUstream_st *stream) {
    g_last_grid_dim = gridDim;
    g_last_block_dim = blockDim;
    g_last_shared_mem = sharedMem;
    g_last_stream = stream;
    return 0;
}

extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                             size_t *sharedMem, void *stream) {
    if (gridDim) *gridDim = g_last_grid_dim;
    if (blockDim) *blockDim = g_last_block_dim;
    if (sharedMem) *sharedMem = g_last_shared_mem;
    if (stream) *reinterpret_cast<void **>(stream) = g_last_stream;
    return cudaSuccess;
}

cudaError_t CUDARTAPI launch(int dst_x, int dst_y, int src_x, int src_y) {
    InterChiplet::launchSync(src_x, src_y, dst_x, dst_y);
    return cudaSuccess;
}

cudaError_t CUDARTAPI waitLaunch(int dst_x, int dst_y, int* src_x, int* src_y) {
    InterChiplet::waitlaunchSync(src_x, src_y, dst_x, dst_y);
    return cudaSuccess;
}

cudaError_t CUDARTAPI barrier(int uid, int src_x, int src_y, int count) {
    InterChiplet::barrierSync(src_x, src_y, uid, count);
    return cudaSuccess;
}

cudaError_t CUDARTAPI lock(int uid, int src_x, int src_y) {
    InterChiplet::lockSync(src_x, src_y, uid);
    return cudaSuccess;
}

cudaError_t CUDARTAPI unlock(int uid, int src_x, int src_y) {
    InterChiplet::unlockSync(src_x, src_y, uid);
    return cudaSuccess;
}

cudaError_t CUDARTAPI sendMessage(int dst_x, int dst_y, int src_x, int src_y, void* addr, int nbyte) {
    if (nbyte <= 0) return cudaSuccess;
    // InterChiplet control/payload buffers used by moe_gpu_worker are host-side
    // C++ vectors/structs.  GPGPU-Sim's cudaMemcpy can report success for these
    // host pointers without actually copying the expected bytes, so keep the
    // protocol path as an explicit host memcpy instead of a CUDA copy.
    std::vector<unsigned char> host(static_cast<size_t>(nbyte));
    std::memcpy(host.data(), addr, static_cast<size_t>(nbyte));
    std::string file_name = InterChiplet::sendSync(src_x, src_y, dst_x, dst_y);
    g_pipe_comm.write_data(file_name.c_str(), host.data(), nbyte);
    g_cycle = InterChiplet::writeSync(g_cycle + 1000, src_x, src_y, dst_x, dst_y, nbyte, 0);
    return cudaSuccess;
}

cudaError_t CUDARTAPI receiveMessage(int dst_x, int dst_y, int src_x, int src_y, void* addr, int nbyte) {
    if (nbyte <= 0) return cudaSuccess;
    std::vector<unsigned char> host(static_cast<size_t>(nbyte));
    std::string file_name = InterChiplet::receiveSync(src_x, src_y, dst_x, dst_y);
    g_pipe_comm.read_data(file_name.c_str(), host.data(), nbyte);
    g_cycle = InterChiplet::readSync(g_cycle + 1000, src_x, src_y, dst_x, dst_y, nbyte, 0);
    std::memcpy(addr, host.data(), static_cast<size_t>(nbyte));
    return cudaSuccess;
}
