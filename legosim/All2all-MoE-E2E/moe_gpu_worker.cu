#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <cstring>

#include "apis_cu.h"
#include "moe_coords.h"

extern "C" void advanceInterchipletCycles(unsigned long long cycles);

#ifdef GPGPUSIM_LEGACY_LAUNCH
extern "C" cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset);
extern "C" cudaError_t CUDARTAPI cudaLaunch(const void *func);
#endif

// Max GPU expert chiplets the all-to-all dispatch matrix can carry. Raised from
// the legacy 4 (matrix[16]) so layouts with up to 128 expert-bearing GPU sites
// drive the NoC. Formal initialization carries the worker's actual outbound
// row and inbound column, not a maximum-sized padded packet.
#define MOE_MAX_NGPU 128

struct GpuCoord { int x; int y; };
struct Header { int32_t ngpu, hidden, inter, elem_bytes; };

// Layout coordinate map from $MOE_COORD_FILE (Task 3). When valid, the ILP's
// per-site (x,y) placement drives every send/receive; otherwise gpu_coord()
// falls back to the legacy hard-coded 4-GPU column so the stock smoke path is
// unchanged. Loaded once in main() and read by gpu_coord().
static moe::CoordMap g_coords;

// Bound the number of heavyweight external GPGPU-Sim processes without
// changing simulated time. Workers retain their InterChiplet state while
// waiting for a host-side slot; only host resource use and wall time change.
static int acquire_external_runner_slot(int &slot_index, int &slot_count) {
    slot_count = 16;
    if (const char *raw = std::getenv("MOE_GPGPUSIM_MAX_PARALLEL")) {
        int parsed = std::atoi(raw);
        if (parsed > 0) slot_count = parsed;
    }
    if (slot_count > MOE_MAX_NGPU) slot_count = MOE_MAX_NGPU;
    for (;;) {
        for (int slot = 0; slot < slot_count; ++slot) {
            std::string path = "/tmp/legosim_gpgpusim_external_slot_" +
                               std::to_string(slot) + ".lock";
            int fd = open(path.c_str(), O_CREAT | O_RDWR, 0666);
            if (fd < 0) continue;
            if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
                slot_index = slot;
                return fd;
            }
            close(fd);
        }
        usleep(10000);
    }
}

__global__ void expert_mlp_kernel(float *a, float *b, float *c, int tokens, int hidden, int inter, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * hidden;
    if (idx < total) {
        float acc = 0.0f;
        // Keep the kernel GPGPU-Sim friendly: avoid integer rem/div emitted by
        // CUDA 12 for modulo expressions, which is both slow and fragile in the
        // old PTX frontend.  The loop still performs real global-memory loads,
        // FP32 FMAs, and a global store for the expert MLP compute envelope.
        float aval = a[idx];
        for (int r = 0; r < repeat; ++r) {
            for (int j = 0; j < inter; ++j) {
                acc = aval * b[j] + acc;
            }
        }
        c[idx] = acc;
    }
}

static void gpu_coord(int id, int &x, int &y) {
    if (g_coords.valid) {
        x = g_coords.gpu[id].x;
        y = g_coords.gpu[id].y;
        return;
    }
    static GpuCoord coords[4] = {{0,0},{0,1},{0,2},{0,3}};
    int cid = (id >= 0 && id < 4) ? id : 0;
    x = coords[cid].x;
    y = coords[cid].y;
}

int main(int argc, char **argv) {
    g_coords = moe::load_coord_env();
    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    if (g_coords.configured && !g_coords.valid) {
        std::cerr << "[GPU " << gpu_id
                  << "] ERROR: MOE_COORD_FILE was set but is invalid"
                  << std::endl;
        return 2;
    }
    if (g_coords.valid &&
        (gpu_id < 0 || gpu_id >= (int)g_coords.gpu.size())) {
        std::cerr << "[GPU " << gpu_id
                  << "] ERROR: worker id is outside coordinate map"
                  << std::endl;
        return 3;
    }
    int cpu_x = argc > 2 ? atoi(argv[2]) : 5;
    int cpu_y = argc > 3 ? atoi(argv[3]) : 5;
    int ffn_repeat = argc > 4 ? atoi(argv[4]) : 2;
    if (ffn_repeat <= 0) {
        std::cerr << "[GPU " << gpu_id << "] invalid ffn_repeat="
                  << ffn_repeat << std::endl;
        return 2;
    }
    // A valid coord file overrides the router coordinate passed on argv so the
    // worker always talks to the ILP-placed router regardless of the yml args.
    if (g_coords.valid) { cpu_x = g_coords.router.x; cpu_y = g_coords.router.y; }
    int me_x, me_y;
    gpu_coord(gpu_id, me_x, me_y);

    int expected_ngpu = g_coords.valid ? g_coords.ngpu : 4;
    if (expected_ngpu <= 0 || expected_ngpu > MOE_MAX_NGPU) {
        std::cerr << "[GPU " << gpu_id << "] invalid coord ngpu="
                  << expected_ngpu << std::endl;
        return 3;
    }
    std::vector<int32_t> init_words(4 + 2 * expected_ngpu, 0);
    receiveMessage(me_x, me_y, cpu_x, cpu_y, init_words.data(),
                   static_cast<int>(init_words.size() * sizeof(int32_t)));
    Header hdr{init_words[0], init_words[1], init_words[2], init_words[3]};
    if (hdr.ngpu != expected_ngpu || hdr.ngpu <= 0 ||
        hdr.ngpu > MOE_MAX_NGPU) {
        std::cerr << "[GPU " << gpu_id << "] init ngpu=" << hdr.ngpu
                  << " differs from coord ngpu=" << expected_ngpu << std::endl;
        return 4;
    }
    std::vector<int32_t> outbound(hdr.ngpu, 0);
    std::vector<int32_t> inbound(hdr.ngpu, 0);
    int compute_tokens = 0;
    for (int i = 0; i < hdr.ngpu; ++i) {
        outbound[i] = init_words[4 + i];
        inbound[i] = init_words[4 + hdr.ngpu + i];
        if (outbound[i] < 0 || inbound[i] < 0) {
            std::cerr << "[GPU " << gpu_id
                      << "] ERROR: negative dispatch token count at peer "
                      << i << std::endl;
            return 5;
        }
        compute_tokens += inbound[i];
    }
    if (compute_tokens <= 0) {
        std::cerr << "[GPU " << gpu_id
                  << "] ERROR: zero global token assignments; refusing "
                     "synthetic one-token compute fallback" << std::endl;
        return 5;
    }

    std::cout << "[GPU " << gpu_id << "] start at (" << me_x << "," << me_y << ") ngpu=" << hdr.ngpu
              << " hidden=" << hdr.hidden << " inter=" << hdr.inter
              << " ffn_repeat=" << ffn_repeat << std::endl;

    // Dispatch phase: ordered source loop avoids all-sender deadlocks.
    std::vector<std::vector<unsigned char>> inbox(hdr.ngpu);
    for (int s = 0; s < hdr.ngpu; ++s) {
        for (int d = 0; d < hdr.ngpu; ++d) {
            if (gpu_id != s && gpu_id != d) continue;
            int tokens = (gpu_id == s) ? outbound[d] : inbound[s];
            int nbytes = tokens * hdr.hidden * hdr.elem_bytes;
            if (tokens <= 0 || nbytes <= 0 || s == d) continue;
            int sx, sy, dx, dy;
            gpu_coord(s, sx, sy); gpu_coord(d, dx, dy);
            if (gpu_id == s) {
                std::vector<unsigned char> buf(nbytes, (unsigned char)((s + d) & 0xff));
                sendMessage(dx, dy, sx, sy, buf.data(), nbytes);
            } else if (gpu_id == d) {
                inbox[s].resize(nbytes);
                receiveMessage(dx, dy, sx, sy, inbox[s].data(), nbytes);
            }
        }
    }

    int local_tokens = outbound[gpu_id];
    int remote_tokens = 0;
    for (int s = 0; s < hdr.ngpu; ++s)
        if (s != gpu_id) remote_tokens += inbound[s];
    if (local_tokens + remote_tokens != compute_tokens) {
        std::cerr << "[GPU " << gpu_id
                  << "] ERROR: inbound dispatch accounting mismatch"
                  << std::endl;
        return 5;
    }

    const char *use_external = getenv("MOE_GPU_EXTERNAL_RUNNER");
    if (!use_external || std::string(use_external) != "0") {
        char exe_path[1024] = {0};
        ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        std::string dir = ".";
        if (n > 0) {
            exe_path[n] = '\0';
            std::string p(exe_path);
            size_t slash = p.find_last_of('/');
            if (slash != std::string::npos) dir = p.substr(0, slash);
        }
        std::string runner = dir + "/moe_gpu_kernel_runner";
        const char *override_runner = getenv("MOE_GPU_KERNEL_RUNNER");
        if (override_runner && override_runner[0]) runner = override_runner;
        // repeat is workload-specific: three matrices for gated Mixtral/fine
        // experts, two for Switch-style FFNs.
        std::string cmd = "\"" + runner + "\" " + std::to_string(compute_tokens) + " " +
                          std::to_string(hdr.hidden) + " " + std::to_string(hdr.inter) +
                          " " + std::to_string(ffn_repeat) + " 2>&1";
        int external_slot = -1;
        int external_slot_count = 0;
        int external_slot_fd = acquire_external_runner_slot(
            external_slot, external_slot_count);
        std::cout << "[GPU " << gpu_id << "] external_gpgpusim_slot="
                  << external_slot << " max_parallel=" << external_slot_count
                  << std::endl;
        std::cout << "[GPU " << gpu_id << "] external_gpgpusim_runner " << cmd << std::endl;
        FILE *runner_pipe = popen(cmd.c_str(), "r");
        if (!runner_pipe) {
            if (external_slot_fd >= 0) {
                flock(external_slot_fd, LOCK_UN);
                close(external_slot_fd);
            }
            std::cerr << "[GPU " << gpu_id << "] failed to start external GPGPU-Sim runner"
                      << std::endl;
            return 10;
        }
        unsigned long long gpu_cycles = 0;
        char line[4096];
        while (fgets(line, sizeof(line), runner_pipe)) {
            std::cout << line;
            const char *marker = std::strstr(line, "gpu_tot_sim_cycle");
            const char *equals = marker ? std::strchr(marker, '=') : nullptr;
            if (equals) {
                char *end = nullptr;
                unsigned long long parsed = std::strtoull(equals + 1, &end, 10);
                if (end != equals + 1) gpu_cycles = parsed;
            }
        }
        int status = pclose(runner_pipe);
        if (external_slot_fd >= 0) {
            flock(external_slot_fd, LOCK_UN);
            close(external_slot_fd);
        }
        int rc = status;
        if (status >= 0 && WIFEXITED(status)) rc = WEXITSTATUS(status);
        else if (status >= 0 && WIFSIGNALED(status)) rc = 128 + WTERMSIG(status);
        std::cout << "[GPU " << gpu_id << "] external_gpgpusim_runner_rc=" << rc << std::endl;
        if (rc != 0) return 10;
        if (gpu_cycles == 0) {
            std::cerr << "[GPU " << gpu_id
                      << "] external runner produced no gpu_tot_sim_cycle" << std::endl;
            return 11;
        }
        advanceInterchipletCycles(gpu_cycles);
        std::cout << "[GPU " << gpu_id << "] interchiplet_compute_cycles="
                  << gpu_cycles << std::endl;
    } else {
        // The in-tree GPGPU-Sim 4.0 runtime is not robust when several simulated
        // GPU processes launch kernels at exactly the same time under one
        // InterChiplet run. Keep workers concurrent for NoC traffic, but
        // serialize only the in-process GPGPU-Sim kernel section.
        int sim_lock_fd = open("/tmp/legosim_gpgpusim_kernel.lock", O_CREAT | O_RDWR, 0666);
        if (sim_lock_fd >= 0) flock(sim_lock_fd, LOCK_EX);

        float *a = nullptr, *b = nullptr, *c = nullptr;
        size_t a_count = (size_t)compute_tokens * hdr.hidden;
        size_t b_count = (size_t)hdr.hidden * hdr.inter;
        std::cout << "[GPU " << gpu_id << "] cuda_alloc_begin compute_tokens=" << compute_tokens
                  << " a_count=" << a_count << " b_count=" << b_count << std::endl;
        cudaMalloc(&a, a_count * sizeof(float));
        cudaMalloc(&b, b_count * sizeof(float));
        cudaMalloc(&c, a_count * sizeof(float));
        std::cout << "[GPU " << gpu_id << "] cuda_alloc_done" << std::endl;
        cudaMemset(a, 1, a_count * sizeof(float));
        cudaMemset(b, 1, b_count * sizeof(float));
        std::cout << "[GPU " << gpu_id << "] cuda_memset_done" << std::endl;
        int threads = 128;
        int blocks = (int)((a_count + threads - 1) / threads);
        std::cout << "[GPU " << gpu_id << "] kernel_launch blocks=" << blocks
                  << " threads=" << threads << std::endl;
#ifdef GPGPUSIM_LEGACY_LAUNCH
        cudaConfigureCall(dim3(blocks), dim3(threads), 0, 0);
        size_t off = 0;
#define PUSH_ARG(v) do { cudaSetupArgument(&(v), sizeof(v), off); off += sizeof(v); } while (0)
        PUSH_ARG(a);
        PUSH_ARG(b);
        PUSH_ARG(c);
        PUSH_ARG(compute_tokens);
        PUSH_ARG(hdr.hidden);
        PUSH_ARG(hdr.inter);
        int repeat = ffn_repeat;
        PUSH_ARG(repeat);
#undef PUSH_ARG
        cudaLaunch((const void*)expert_mlp_kernel);
#else
        expert_mlp_kernel<<<blocks, threads>>>(a, b, c, compute_tokens,
                                               hdr.hidden, hdr.inter,
                                               ffn_repeat);
#endif
        cudaDeviceSynchronize();
        std::cout << "[GPU " << gpu_id << "] kernel_done" << std::endl;
        cudaFree(a); cudaFree(b); cudaFree(c);
        if (sim_lock_fd >= 0) { flock(sim_lock_fd, LOCK_UN); close(sim_lock_fd); }
    }

    // Combine phase: expert destination returns hidden states to original source GPUs.
    for (int s = 0; s < hdr.ngpu; ++s) {
        for (int d = 0; d < hdr.ngpu; ++d) {
            if (gpu_id != s && gpu_id != d) continue;
            int tokens = (gpu_id == s) ? outbound[d] : inbound[s];
            int nbytes = tokens * hdr.hidden * hdr.elem_bytes;
            if (tokens <= 0 || nbytes <= 0 || s == d) continue;
            int sx, sy, dx, dy;
            gpu_coord(s, sx, sy); gpu_coord(d, dx, dy);
            if (gpu_id == d) {
                std::vector<unsigned char> buf(nbytes, 7);
                sendMessage(sx, sy, dx, dy, buf.data(), nbytes);
            } else if (gpu_id == s) {
                std::vector<unsigned char> buf(nbytes);
                receiveMessage(sx, sy, dx, dy, buf.data(), nbytes);
            }
        }
    }

    sendMessage(cpu_x, cpu_y, me_x, me_y, &remote_tokens, sizeof(int32_t));
    std::cout << "[GPU " << gpu_id << "] done local_tokens=" << local_tokens
              << " remote_tokens=" << remote_tokens << std::endl;
    return 0;
}
