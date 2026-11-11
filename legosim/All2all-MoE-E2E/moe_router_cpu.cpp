#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "pipe_comm.h"
#include "sync_protocol.h"
#include "moe_coords.h"

// Router and workers share a four-int header followed by the destination
// worker's outbound row and inbound column (2*ngpu entries). This is all a
// worker needs from the global dispatch matrix and keeps control traffic O(E^2)
// across the system instead of replicating an E^2 matrix to E workers.
#define MOE_MAX_NGPU 128

struct GpuCoord { int x; int y; };

static InterChiplet::PipeComm g_pipe_comm;
static InterChiplet::TimeType g_cycle = 0;

static void router_send_message(int dst_x, int dst_y, int src_x, int src_y, void *addr, int nbyte) {
    if (nbyte <= 0) return;
    std::string file_name = InterChiplet::sendSync(src_x, src_y, dst_x, dst_y);
    g_pipe_comm.write_data(file_name.c_str(), addr, nbyte);
    g_cycle = InterChiplet::writeSync(g_cycle + 1000, src_x, src_y, dst_x, dst_y, nbyte, 0);
}

static void router_receive_message(int dst_x, int dst_y, int src_x, int src_y, void *addr, int nbyte) {
    if (nbyte <= 0) return;
    std::string file_name = InterChiplet::receiveSync(src_x, src_y, dst_x, dst_y);
    g_pipe_comm.read_data(file_name.c_str(), addr, nbyte);
    g_cycle = InterChiplet::readSync(g_cycle + 1000, src_x, src_y, dst_x, dst_y, nbyte, 0);
}

// Load round 0 of a dispatch_matrix.txt file written by
// real_trace_to_dispatch_matrix.py. Returns true on success and fills
// `dispatch` (ngpu*ngpu int32 row-major). On any parse error returns false
// and leaves `dispatch` untouched so the caller can fall back to Zipf.
//
// File format (text, line-oriented):
//   # comment lines start with '#'
//   ngpu <N>
//   experts <E>            (ignored here; informational)
//   topk <K>                (ignored here; informational)
//   n_rounds <R>
//   round <id> layer <L|-1|all> phase <prefill|decode|all>
//   <r0c0> <r0c1> ... <r0cN-1>
//   ...
static bool load_dispatch_matrix_from_file(const std::string &path,
                                           int expected_ngpu,
                                           std::vector<int32_t> &dispatch) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[CPU router] WARN: dispatch file open failed: " << path
                  << " -- falling back to Zipf" << std::endl;
        return false;
    }
    int file_ngpu = -1;
    int file_rounds = 0;
    std::string line;
    while (std::getline(fin, line)) {
        // Trim
        auto not_ws = [](char c){ return !std::isspace((unsigned char)c); };
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), not_ws));
        line.erase(std::find_if(line.rbegin(), line.rend(), not_ws).base(), line.end());
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key; iss >> key;
        if (key == "ngpu")      { iss >> file_ngpu; }
        else if (key == "experts" || key == "topk") { /* informational */ }
        else if (key == "n_rounds") { iss >> file_rounds; }
        else if (key == "round") {
            // Found a round header; the next ngpu non-blank/non-comment lines are the matrix.
            // Path A semantics: load round 0 and stop.
            if (file_ngpu <= 0) {
                std::cerr << "[CPU router] ERROR: ngpu missing before round in " << path << std::endl;
                return false;
            }
            if (file_ngpu != expected_ngpu) {
                std::cerr << "[CPU router] WARN: dispatch file ngpu=" << file_ngpu
                          << " but C++ ngpu=" << expected_ngpu << " -- truncating/padding" << std::endl;
            }
            int n = file_ngpu;
            std::vector<int32_t> mat(n * n, 0);
            int rows_read = 0;
            while (rows_read < n && std::getline(fin, line)) {
                auto not_ws2 = [](char c){ return !std::isspace((unsigned char)c); };
                line.erase(line.begin(), std::find_if(line.begin(), line.end(), not_ws2));
                line.erase(std::find_if(line.rbegin(), line.rend(), not_ws2).base(), line.end());
                if (line.empty() || line[0] == '#') continue;
                std::istringstream rs(line);
                for (int c = 0; c < n; ++c) rs >> mat[rows_read * n + c];
                ++rows_read;
            }
            if (rows_read != n) {
                std::cerr << "[CPU router] ERROR: round 0 had " << rows_read << " rows, expected " << n << std::endl;
                return false;
            }
            // Project to expected_ngpu (top-left submatrix; aggregate any extra rows/cols into the last bucket).
            std::vector<int32_t> proj(expected_ngpu * expected_ngpu, 0);
            for (int r = 0; r < n; ++r) {
                int dr = std::min(r, expected_ngpu - 1);
                for (int c = 0; c < n; ++c) {
                    int dc = std::min(c, expected_ngpu - 1);
                    proj[dr * expected_ngpu + dc] += mat[r * n + c];
                }
            }
            dispatch = std::move(proj);
            std::cout << "[CPU router] loaded dispatch matrix from " << path;
            std::cout << " (file_ngpu=" << n;
            std::cout << " rounds_in_file=" << file_rounds;
            std::cout << " using round 0)" << std::endl;
            return true;
        }
    }
    std::cerr << "[CPU router] ERROR: no 'round' header found in " << path << std::endl;
    return false;
}

static std::vector<double> zipf_weights(int n, double alpha, int shift) {
    std::vector<double> w(n, 0.0);
    for (int r = 0; r < n; ++r) w[(r + shift) % n] = 1.0 / std::pow(double(r + 1), alpha);
    double s = std::accumulate(w.begin(), w.end(), 0.0);
    for (double &v : w) v /= s;
    return w;
}

static std::vector<int> sample_topk(std::mt19937 &rng, const std::vector<double> &w, int k) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<std::pair<double,int>> keys;
    keys.reserve(w.size());
    for (int i = 0; i < (int)w.size(); ++i) {
        if (w[i] <= 0) continue;
        keys.push_back({std::pow(uni(rng), 1.0 / w[i]), i});
    }
    std::sort(keys.begin(), keys.end(), std::greater<std::pair<double,int>>());
    std::vector<int> out;
    for (int i = 0; i < k && i < (int)keys.size(); ++i) out.push_back(keys[i].second);
    return out;
}

static int expert_to_gpu(int expert, int experts, int ngpu) {
    int per = (experts + ngpu - 1) / ngpu;
    return std::min(ngpu - 1, expert / per);
}

int main(int argc, char **argv) {
    // Task 3: optional ILP layout coordinate map. When $MOE_COORD_FILE is a
    // valid file the router places itself and every GPU worker at the
    // ILP-chosen (x,y). An unset variable retains the legacy smoke behavior;
    // an explicitly configured but invalid file is a formal input error.
    moe::CoordMap coords = moe::load_coord_env();
    if (coords.configured && !coords.valid) {
        std::cerr << "[CPU router] ERROR: MOE_COORD_FILE was set but is invalid"
                  << std::endl;
        return 2;
    }

    int cpu_x = argc > 1 ? std::atoi(argv[1]) : 5;
    int cpu_y = argc > 2 ? std::atoi(argv[2]) : 5;
    int ngpu = argc > 3 ? std::atoi(argv[3]) : 4;
    int tokens_per_gpu = argc > 4 ? std::atoi(argv[4]) : 4;
    int experts = argc > 5 ? std::atoi(argv[5]) : 256;
    int topk = argc > 6 ? std::atoi(argv[6]) : 8;
    int hidden = argc > 7 ? std::atoi(argv[7]) : 128;   // smoke-scaled hidden dim for GPGPU-Sim
    int inter = argc > 8 ? std::atoi(argv[8]) : 64;     // smoke-scaled expert FFN dim
    double alpha = argc > 9 ? std::atof(argv[9]) : 1.2;
    int hot_shift = argc > 10 ? std::atoi(argv[10]) : 0;
    int seed = argc > 11 ? std::atoi(argv[11]) : 202501;
    // P1: number of dedicated CiM expert chiplets to drive (0 = original
    // all-digital behavior, unchanged). CiM nodes live on column x=1.
    int ncim = argc > 12 ? std::atoi(argv[12]) : 0;
    int cold_tokens_per_cim = argc > 13 ? std::atoi(argv[13]) : 4;  // placeholder; P2 = real cold split
    std::vector<GpuCoord> cim = {{1, 0}, {1, 1}, {1, 2}, {1, 3}};
    if (ncim > (int)cim.size()) ncim = cim.size();

    std::vector<GpuCoord> gpu = {{0,0}, {0,1}, {0,2}, {0,3}};
    // Override with the ILP layout when a coord file is present. The router
    // coordinate and the full GPU (x,y) list come from the file; ngpu is taken
    // from the file's declared count (still honoring the argv upper bound if it
    // was smaller, to stay compatible with per-run experiments).
    if (coords.valid) {
        cpu_x = coords.router.x;
        cpu_y = coords.router.y;
        gpu.clear();
        for (const auto &c : coords.gpu) gpu.push_back(GpuCoord{c.x, c.y});
        if (ngpu > (int)gpu.size()) ngpu = (int)gpu.size();
        if (coords.ngpu > 0 && coords.ngpu <= (int)gpu.size()) ngpu = coords.ngpu;
        std::cout << "[CPU router] using ILP layout coords: router=(" << cpu_x << ","
                  << cpu_y << ") ngpu=" << ngpu << std::endl;
    }
    if (ngpu > (int)gpu.size()) ngpu = gpu.size();
    if (ngpu > MOE_MAX_NGPU) {
        std::cerr << "[CPU router] ERROR: ngpu=" << ngpu
                  << " exceeds formal ABI limit " << MOE_MAX_NGPU
                  << std::endl;
        return 2;
    }

    // ---- P2: per-expert hot/cold classification by Zipf popularity ----
    // tau = hot fraction (env MOE_HOT_FRACTION, default 0.25). The hottest
    // ceil(tau*experts) experts are HOT -> digital GPU chiplets; the rest are
    // COLD -> dedicated CiM chiplets. tau is the Pareto sweep knob (P5).
    double hot_fraction = 0.25;
    if (const char *hf = std::getenv("MOE_HOT_FRACTION")) { if (*hf) hot_fraction = std::atof(hf); }
    if (hot_fraction < 0.0) hot_fraction = 0.0;
    if (hot_fraction > 1.0) hot_fraction = 1.0;
    std::vector<double> hotness = zipf_weights(experts, alpha, hot_shift);
    std::vector<int> by_hot(experts);
    std::iota(by_hot.begin(), by_hot.end(), 0);
    std::sort(by_hot.begin(), by_hot.end(), [&](int a, int b) { return hotness[a] > hotness[b]; });
    int n_hot = (int)std::ceil(hot_fraction * experts);
    if (n_hot > experts) n_hot = experts;
    int n_cold = experts - n_hot;
    std::vector<char> is_hot(experts, 0);
    for (int i = 0; i < n_hot; ++i) is_hot[by_hot[i]] = 1;
    // Weight cache: the top MOE_CACHE_EXPERTS hottest experts are assumed
    // resident in the digital chiplet's weight cache (no DRAM fetch). Digital
    // tokens hitting experts OUTSIDE the cache pay a weight-fetch penalty (P5);
    // CiM experts are resident in-array (never fetched).
    int cache_experts = 32;
    if (const char *ce = std::getenv("MOE_CACHE_EXPERTS")) { if (*ce) cache_experts = std::atoi(ce); }
    if (cache_experts > experts) cache_experts = experts;
    if (cache_experts < 0) cache_experts = 0;
    std::vector<char> is_cached(experts, 0);
    for (int i = 0; i < cache_experts; ++i) is_cached[by_hot[i]] = 1;
    long digital_uncached = 0;
    // cold experts -> CiM nodes, evenly partitioned by cold rank
    std::vector<int> cold_cnode(experts, -1);
    if (ncim > 0 && n_cold > 0) {
        for (int i = n_hot, ci = 0; i < experts; ++i, ++ci)
            cold_cnode[by_hot[i]] = (ci * ncim) / n_cold;
    }
    int ncim_eff = ncim > 0 ? ncim : 1;
    std::vector<int32_t> dispatch_cim(ngpu * ncim_eff, 0);
    std::vector<int32_t> cim_tokens(ncim_eff, 0);

    std::vector<int32_t> dispatch(ngpu * ngpu, 0);
    std::vector<int32_t> expert_tokens(ngpu, 0);

    // Real-trace path: if MOE_DISPATCH_FILE is set to a readable file, load the
    // (round 0) dispatch matrix from it and skip Zipf sampling. Otherwise fall
    // back to the original synthetic Zipf top-K sampler so existing yml's work
    // unchanged.
    const char *dispatch_env = std::getenv("MOE_DISPATCH_FILE");
    bool loaded = false;
    if (dispatch_env && *dispatch_env) {
        loaded = load_dispatch_matrix_from_file(std::string(dispatch_env), ngpu, dispatch);
        if (loaded) {
            for (int d = 0; d < ngpu; ++d) {
                int sum = 0;
                for (int s = 0; s < ngpu; ++s) sum += dispatch[s * ngpu + d];
                expert_tokens[d] = sum;
            }
            std::cout << "[CPU router] using REAL MoE dispatch matrix (Zipf sampling skipped)" << std::endl;
        } else {
            std::cerr << "[CPU router] WARN: dispatch file path was set but load failed; using Zipf fallback" << std::endl;
        }
    }
    if (!loaded) {
        std::mt19937 rng(seed);
        auto weights = zipf_weights(experts, alpha, hot_shift);
        for (int src = 0; src < ngpu; ++src) {
            for (int t = 0; t < tokens_per_gpu; ++t) {
                auto chosen = sample_topk(rng, weights, topk);
                for (int e : chosen) {
                    if (is_hot[e] || ncim == 0) {
                        int dst = expert_to_gpu(e, experts, ngpu);   // HOT -> digital GPU chiplet
                        dispatch[src * ngpu + dst] += 1;
                        expert_tokens[dst] += 1;
                        if (!is_cached[e]) ++digital_uncached;       // DRAM weight-fetch
                    } else {
                        int c = cold_cnode[e];                       // COLD -> dedicated CiM chiplet
                        dispatch_cim[src * ncim + c] += 1;
                        cim_tokens[c] += 1;
                    }
                }
            }
        }
    }

    std::cout << "[CPU router] DeepSeek-like MoE E2E smoke" << std::endl;
    std::cout << "trace_source=" << (loaded ? "real_dispatch_file" : "zipf_synthetic") << std::endl;
    std::cout << "ngpu=" << ngpu << " tokens_per_gpu=" << tokens_per_gpu
              << " experts=" << experts << " topk=" << topk
              << " hidden=" << hidden << " inter=" << inter
              << " alpha=" << alpha << " hot_shift=" << hot_shift << " seed=" << seed << std::endl;
    std::cout << "dispatch_matrix" << std::endl;
    for (int s = 0; s < ngpu; ++s) {
        std::cout << "src " << s;
        for (int d = 0; d < ngpu; ++d) std::cout << " " << dispatch[s * ngpu + d];
        std::cout << std::endl;
    }
    for (int d = 0; d < ngpu; ++d) {
        std::cout << "expert_gpu " << d << " token_assignments=" << expert_tokens[d] << std::endl;
    }
    // P2: hot/cold split summary + CiM dispatch matrix (print-only validation)
    std::cout << "hot_fraction=" << hot_fraction << " n_hot_experts=" << n_hot
              << " n_cold_experts=" << n_cold << " ncim=" << ncim << std::endl;
    std::cout << "cache_experts=" << cache_experts << " digital_tokens_uncached=" << digital_uncached << std::endl;
    if (ncim > 0) {
        std::cout << "digital_nodes(x,y)=";
        for (int d = 0; d < ngpu; ++d) std::cout << " (" << gpu[d].x << "," << gpu[d].y << ")";
        std::cout << "  cim_nodes(x,y)=";
        for (int c = 0; c < ncim; ++c) std::cout << " (" << cim[c].x << "," << cim[c].y << ")";
        std::cout << std::endl;
        std::cout << "dispatch_cim_matrix" << std::endl;
        for (int s = 0; s < ngpu; ++s) {
            std::cout << "src " << s;
            for (int c = 0; c < ncim; ++c) std::cout << " " << dispatch_cim[s * ncim + c];
            std::cout << std::endl;
        }
        long tot_cold = 0, tot_hot = 0;
        for (int c = 0; c < ncim; ++c) {
            std::cout << "cim_node " << c << " cold_token_assignments=" << cim_tokens[c] << std::endl;
            tot_cold += cim_tokens[c];
        }
        for (int d = 0; d < ngpu; ++d) tot_hot += expert_tokens[d];
        std::cout << "tokens_hot_digital=" << tot_hot << " tokens_cold_cim=" << tot_cold << std::endl;
    }

    std::vector<int32_t> gpu_init(4 + 2 * ngpu, 0);
    gpu_init[0] = ngpu;
    gpu_init[1] = hidden;
    gpu_init[2] = inter;
    gpu_init[3] = 2;
    int gpu_init_bytes = static_cast<int>(gpu_init.size() * sizeof(int32_t));
    std::cout << "[CPU router] gpu_init_protocol=row_col init_bytes_per_worker="
              << gpu_init_bytes << std::endl;
    for (int i = 0; i < ngpu; ++i) {
        for (int d = 0; d < ngpu; ++d)
            gpu_init[4 + d] = dispatch[i * ngpu + d];
        for (int s = 0; s < ngpu; ++s)
            gpu_init[4 + ngpu + s] = dispatch[s * ngpu + i];
        // A single dynamically-sized packet avoids both back-to-back endpoint
        // deadlocks and replication of the entire global matrix.
        router_send_message(gpu[i].x, gpu[i].y, cpu_x, cpu_y,
                            gpu_init.data(), gpu_init_bytes);
    }

    // P2: dispatch the REAL per-node cold-token workload to each CiM chiplet
    // (counts = column sums of dispatch_cim). The P1 placeholder is gone.
    (void)cold_tokens_per_cim;
    struct CimHeader { int32_t cold_tokens, hidden, inter, elem_bytes; };
    for (int i = 0; i < ncim; ++i) {
        int32_t cold_i = cim_tokens[i];
        CimHeader ch{cold_i, hidden, inter, 2};
        router_send_message(cim[i].x, cim[i].y, cpu_x, cpu_y, &ch, sizeof(ch));
        int bytes_i = cold_i * hidden * 2;
        if (bytes_i > 0) {
            std::vector<unsigned char> buf(bytes_i, (unsigned char)(i & 0xff));
            router_send_message(cim[i].x, cim[i].y, cpu_x, cpu_y, buf.data(), bytes_i);
        }
    }

    std::vector<int32_t> summaries(ngpu, 0);
    for (int i = 0; i < ngpu; ++i) {
        router_receive_message(cpu_x, cpu_y, gpu[i].x, gpu[i].y, &summaries[i], sizeof(int32_t));
        std::cout << "gpu " << i << " completed_remote_tokens=" << summaries[i] << std::endl;
    }
    std::vector<int32_t> cim_summaries(ncim, 0);
    for (int i = 0; i < ncim; ++i) {
        int bytes_i = cim_tokens[i] * hidden * 2;
        if (bytes_i > 0) {
            std::vector<unsigned char> buf(bytes_i);
            router_receive_message(cpu_x, cpu_y, cim[i].x, cim[i].y, buf.data(), bytes_i);
        }
        router_receive_message(cpu_x, cpu_y, cim[i].x, cim[i].y, &cim_summaries[i], sizeof(int32_t));
        std::cout << "cim " << i << " completed_cold_tokens=" << cim_summaries[i] << std::endl;
    }
    std::cout << "ncim=" << ncim << " split=per_expert_zipf hot_fraction=" << hot_fraction << std::endl;
    // Report the final end-to-end logical time to Interchiplet. CYCLE is a
    // one-way command; sendCycleCmd deliberately does not wait for a reply.
    // g_cycle now includes the completion messages whose worker timestamps
    // include dispatch, measured GPGPU-Sim compute, combine, and NoC delay.
    InterChiplet::sendCycleCmd(g_cycle);
    std::cout << "[CPU router] done" << std::endl;
    return 0;
}
