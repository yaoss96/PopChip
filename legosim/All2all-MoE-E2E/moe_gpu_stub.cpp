// moe_gpu_stub.cpp -- plain-C++ stand-in for moe_gpu_worker (NO CUDA / NO GPGPU-Sim).
//
// Replicates moe_gpu_worker's inter-chiplet message protocol (dispatch all-to-all,
// combine all-to-all, summary) but skips the CUDA expert kernel. Lets the full
// heterogeneous co-simulation (router + digital + CiM + popnet) run on a plain
// x86_64 host for validation when GPGPU-Sim is not available. Not for timing
// fidelity of the digital path -- use the real moe_gpu_worker in WSL for that.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "pipe_comm.h"
#include "sync_protocol.h"
#include "moe_coords.h"

// Must match moe_router_cpu.cpp / moe_gpu_worker.cu dynamic init layout.
#define MOE_MAX_NGPU 128

struct GpuCoord { int x; int y; };
struct Header { int32_t ngpu, hidden, inter, elem_bytes; };

static moe::CoordMap g_coords;

static InterChiplet::PipeComm g_pipe;
static InterChiplet::TimeType g_cycle = 0;

static void send_msg(int dst_x, int dst_y, int src_x, int src_y, void *addr, int nbyte) {
    if (nbyte <= 0) return;
    std::string fn = InterChiplet::sendSync(src_x, src_y, dst_x, dst_y);
    g_pipe.write_data(fn.c_str(), addr, nbyte);
    g_cycle = InterChiplet::writeSync(g_cycle + 1000, src_x, src_y, dst_x, dst_y, nbyte, 0);
}
static void recv_msg(int dst_x, int dst_y, int src_x, int src_y, void *addr, int nbyte) {
    if (nbyte <= 0) return;
    std::string fn = InterChiplet::receiveSync(src_x, src_y, dst_x, dst_y);
    g_pipe.read_data(fn.c_str(), addr, nbyte);
    g_cycle = InterChiplet::readSync(g_cycle + 1000, src_x, src_y, dst_x, dst_y, nbyte, 0);
}
static void gpu_coord(int id, int &x, int &y) {
    if (g_coords.valid) {
        x = g_coords.gpu[id].x; y = g_coords.gpu[id].y; return;
    }
    static GpuCoord coords[4] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}};
    int cid = (id >= 0 && id < 4) ? id : 0;
    x = coords[cid].x; y = coords[cid].y;
}


int main(int argc, char **argv) {
    g_coords = moe::load_coord_env();
    int gpu_id = argc > 1 ? std::atoi(argv[1]) : 0;
    if (g_coords.configured && !g_coords.valid) {
        std::cerr << "[GPU stub " << gpu_id
                  << "] ERROR: MOE_COORD_FILE was set but is invalid"
                  << std::endl;
        return 2;
    }
    if (g_coords.valid &&
        (gpu_id < 0 || gpu_id >= (int)g_coords.gpu.size())) {
        std::cerr << "[GPU stub " << gpu_id
                  << "] ERROR: worker id is outside coordinate map"
                  << std::endl;
        return 3;
    }
    int cpu_x = argc > 2 ? std::atoi(argv[2]) : 5;
    int cpu_y = argc > 3 ? std::atoi(argv[3]) : 5;
    if (g_coords.valid) { cpu_x = g_coords.router.x; cpu_y = g_coords.router.y; }
    int me_x, me_y; gpu_coord(gpu_id, me_x, me_y);

    int expected_ngpu = g_coords.valid ? g_coords.ngpu : 4;
    if (expected_ngpu <= 0 || expected_ngpu > MOE_MAX_NGPU) return 3;
    std::vector<int32_t> init_words(4 + 2 * expected_ngpu, 0);
    recv_msg(me_x, me_y, cpu_x, cpu_y, init_words.data(),
             static_cast<int>(init_words.size() * sizeof(int32_t)));
    Header hdr{init_words[0], init_words[1], init_words[2], init_words[3]};
    if (hdr.ngpu != expected_ngpu || hdr.ngpu <= 0 ||
        hdr.ngpu > MOE_MAX_NGPU) return 4;
    std::vector<int32_t> outbound(hdr.ngpu, 0);
    std::vector<int32_t> inbound(hdr.ngpu, 0);
    int compute_tokens = 0;
    for (int i = 0; i < hdr.ngpu; ++i) {
        outbound[i] = init_words[4 + i];
        inbound[i] = init_words[4 + hdr.ngpu + i];
        if (outbound[i] < 0 || inbound[i] < 0) {
            std::cerr << "[GPU stub " << gpu_id
                      << "] ERROR: negative dispatch token count at peer "
                      << i << std::endl;
            return 5;
        }
        compute_tokens += inbound[i];
    }
    if (compute_tokens <= 0) {
        std::cerr << "[GPU stub " << gpu_id
                  << "] ERROR: zero global token assignments; refusing "
                     "synthetic one-token compute fallback" << std::endl;
        return 5;
    }

    // Dispatch all-to-all (ordered source loop avoids all-sender deadlocks).
    std::vector<std::vector<unsigned char>> inbox(hdr.ngpu);
    for (int s = 0; s < hdr.ngpu; ++s) {
        for (int d = 0; d < hdr.ngpu; ++d) {
            if (gpu_id != s && gpu_id != d) continue;
            int tokens = (gpu_id == s) ? outbound[d] : inbound[s];
            int nbytes = tokens * hdr.hidden * hdr.elem_bytes;
            if (tokens <= 0 || nbytes <= 0 || s == d) continue;
            int sx, sy, dx, dy; gpu_coord(s, sx, sy); gpu_coord(d, dx, dy);
            if (gpu_id == s) {
                std::vector<unsigned char> buf(nbytes, (unsigned char)((s + d) & 0xff));
                send_msg(dx, dy, sx, sy, buf.data(), nbytes);
            } else if (gpu_id == d) {
                inbox[s].resize(nbytes);
                recv_msg(dx, dy, sx, sy, inbox[s].data(), nbytes);
            }
        }
    }

    int local_tokens = outbound[gpu_id];
    int remote_tokens = 0;
    for (int s = 0; s < hdr.ngpu; ++s)
        if (s != gpu_id) remote_tokens += inbound[s];
    if (local_tokens + remote_tokens != compute_tokens) {
        std::cerr << "[GPU stub " << gpu_id
                  << "] ERROR: inbound dispatch accounting mismatch"
                  << std::endl;
        return 5;
    }
    // (expert MLP kernel skipped in the stub)

    // Combine all-to-all (expert destination returns hidden states to sources).
    for (int s = 0; s < hdr.ngpu; ++s) {
        for (int d = 0; d < hdr.ngpu; ++d) {
            if (gpu_id != s && gpu_id != d) continue;
            int tokens = (gpu_id == s) ? outbound[d] : inbound[s];
            int nbytes = tokens * hdr.hidden * hdr.elem_bytes;
            if (tokens <= 0 || nbytes <= 0 || s == d) continue;
            int sx, sy, dx, dy; gpu_coord(s, sx, sy); gpu_coord(d, dx, dy);
            if (gpu_id == d) {
                std::vector<unsigned char> buf(nbytes, 7);
                send_msg(sx, sy, dx, dy, buf.data(), nbytes);
            } else if (gpu_id == s) {
                std::vector<unsigned char> buf(nbytes);
                recv_msg(sx, sy, dx, dy, buf.data(), nbytes);
            }
        }
    }

    send_msg(cpu_x, cpu_y, me_x, me_y, &remote_tokens, sizeof(int32_t));
    std::cout << "[GPU stub " << gpu_id << "] done local=" << local_tokens
              << " remote=" << remote_tokens
              << " compute_tokens=" << compute_tokens << std::endl;
    return 0;
}
