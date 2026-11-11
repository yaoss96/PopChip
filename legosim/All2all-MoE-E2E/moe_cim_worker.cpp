// moe_cim_worker.cpp -- dedicated CiM expert chiplet (P3: real MNSIM numbers).
//
// A non-CUDA chiplet process that joins the LEGOSim co-simulation like
// moe_gpu_worker, representing a Compute-in-Memory (ReRAM) expert tile.
// Its compute latency/energy come from REAL MNSIM 2.0 characterization
// (cim_backend/moe_expert_mnsim.py), not a hand-fit model:
//
//   mode B (default): look up per-token latency_ns/energy_nJ for the expert
//                     FFN shape (hidden x inter) in cim_char_table.json, then
//                     scale linearly by the cold-token count.
//   mode A (online) : invoke the MNSIM driver live (popen) for fresh numbers.
//
// Knobs (env): MOE_CIM_MODE=table|online, MOE_CIM_GHZ (ns->cycle, default 1.0),
//   MOE_CIM_DRIVER, MOE_CIM_MNSIM_DIR, MOE_CIM_SIMCONFIG, MOE_CIM_PYTHON,
//   MOE_CIM_FIXED_NS.
// Placement: CiM nodes on column x=1 (nodes 1,7,13,19), one hop from the GPUs.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pipe_comm.h"
#include "sync_protocol.h"

struct CimCoord { int x; int y; };
struct CimHeader { int32_t cold_tokens; int32_t hidden; int32_t inter; int32_t elem_bytes; };

static InterChiplet::PipeComm g_pipe_comm;
static InterChiplet::TimeType g_cycle = 0;

static void cim_send_message(int dst_x, int dst_y, int src_x, int src_y, void *addr, int nbyte) {
    if (nbyte <= 0) return;
    std::string file_name = InterChiplet::sendSync(src_x, src_y, dst_x, dst_y);
    g_pipe_comm.write_data(file_name.c_str(), addr, nbyte);
    g_cycle = InterChiplet::writeSync(g_cycle + 1000, src_x, src_y, dst_x, dst_y, nbyte, 0);
}

static void cim_receive_message(int dst_x, int dst_y, int src_x, int src_y, void *addr, int nbyte) {
    if (nbyte <= 0) return;
    std::string file_name = InterChiplet::receiveSync(src_x, src_y, dst_x, dst_y);
    g_pipe_comm.read_data(file_name.c_str(), addr, nbyte);
    g_cycle = InterChiplet::readSync(g_cycle + 1000, src_x, src_y, dst_x, dst_y, nbyte, 0);
}

static void cim_coord(int id, int &x, int &y) {
    static CimCoord coords[4] = {{1, 0}, {1, 1}, {1, 2}, {1, 3}};
    int i = id % 4;
    x = coords[i].x;
    y = coords[i].y;
}

// Extract a numeric JSON field that appears after position `from` in `s`.
static bool grab_field(const std::string &s, const std::string &field, size_t from, double &out) {
    size_t p = s.find("\"" + field + "\"", from);
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    out = std::atof(s.c_str() + p + 1);
    return true;
}

// Mode B: per-token latency/energy from cim_char_table.json keyed "<hidden>x<inter>".
static bool lookup_cim_char(const std::string &path, int hidden, int inter,
                            double &lat_ns, double &eng_nJ) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::string key = "\"" + std::to_string(hidden) + "x" + std::to_string(inter) + "\"";
    size_t k = s.find(key);
    if (k == std::string::npos) return false;
    return grab_field(s, "latency_ns_per_token", k, lat_ns) &&
           grab_field(s, "energy_nJ_per_token", k, eng_nJ);
}

// Mode A: invoke the MNSIM driver live and parse its JSON line.
static bool online_cim_char(const std::string &python, const std::string &mnsim_dir,
                            const std::string &driver, const std::string &sim_cfg,
                            int hidden, int inter, double &lat_ns, double &eng_nJ) {
    std::stringstream cmd;
    cmd << "cd '" << mnsim_dir << "' && PYTHONPATH='" << mnsim_dir << "' '" << python << "' '" << driver
        << "' --sim-config '" << sim_cfg << "' --hidden " << hidden << " --inter " << inter
        << " 2>/dev/null";
    FILE *pp = popen(cmd.str().c_str(), "r");
    if (!pp) return false;
    std::string out;
    char buf[512];
    while (fgets(buf, sizeof(buf), pp)) out += buf;
    pclose(pp);
    size_t k = out.find("latency_ns_per_token");
    if (k == std::string::npos) return false;
    return grab_field(out, "latency_ns_per_token", 0, lat_ns) &&
           grab_field(out, "energy_nJ_per_token", 0, eng_nJ);
}

static std::string env_or(const char *name, const std::string &def) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : def;
}

int main(int argc, char **argv) {
    int cim_id = argc > 1 ? std::atoi(argv[1]) : 0;
    int cpu_x = argc > 2 ? std::atoi(argv[2]) : 5;
    int cpu_y = argc > 3 ? std::atoi(argv[3]) : 5;
    std::string char_table = argc > 4 ? std::string(argv[4]) : env_or("MOE_CIM_CHAR_TABLE", "");

    std::string mode = env_or("MOE_CIM_MODE", "table");
    double ghz = std::atof(env_or("MOE_CIM_GHZ", "1.0").c_str());     // ns -> cycles
    if (ghz <= 0) ghz = 1.0;
    double fixed_ns = std::atof(env_or("MOE_CIM_FIXED_NS", "5000").c_str());  // fallback per token

    int me_x, me_y;
    cim_coord(cim_id, me_x, me_y);

    // 1) Receive the cold-expert workload header from the CPU router.
    CimHeader hdr;
    cim_receive_message(me_x, me_y, cpu_x, cpu_y, &hdr, sizeof(hdr));
    int cold = hdr.cold_tokens > 0 ? hdr.cold_tokens : 0;

    // 2) Receive the cold-token hidden-state payload (real NoC dispatch traffic).
    int payload_bytes = cold * hdr.hidden * hdr.elem_bytes;
    if (payload_bytes > 0) {
        std::vector<unsigned char> inbox(payload_bytes);
        cim_receive_message(me_x, me_y, cpu_x, cpu_y, inbox.data(), payload_bytes);
    }

    // 3) Real MNSIM compute cost for this expert FFN shape, scaled by token count.
    double lat_ns = 0.0, eng_nJ = 0.0;
    std::string backend = "idle";
    if (cold > 0) {                  // skip the MNSIM call entirely for an idle CiM node
        bool ok = false;
        if (mode == "online") {      // mode A: invoke real MNSIM live, per round
            ok = online_cim_char(env_or("MOE_CIM_PYTHON", "python3"),
                                 env_or("MOE_CIM_MNSIM_DIR", "."),
                                 env_or("MOE_CIM_DRIVER", "cim_backend/moe_expert_mnsim.py"),
                                 env_or("MOE_CIM_SIMCONFIG", "SimConfig.ini"),
                                 hdr.hidden, hdr.inter, lat_ns, eng_nJ);
            backend = ok ? "mnsim_online" : "";
        }
        if (!ok && !char_table.empty()) {   // mode B: real-MNSIM characterization table
            ok = lookup_cim_char(char_table, hdr.hidden, hdr.inter, lat_ns, eng_nJ);
            backend = ok ? "mnsim_table" : "";
        }
        if (!ok) {                   // graceful fallback so the co-sim still runs
            lat_ns = fixed_ns;
            eng_nJ = 0.0;
            backend = "fixed_fallback";
        }
    }

    long compute_cycles = (long)std::llround(lat_ns * (double)cold * ghz);
    double total_energy_nJ = eng_nJ * (double)cold;
    g_cycle += compute_cycles;

    std::cout << "[CiM " << cim_id << "] at (" << me_x << "," << me_y << ") backend=" << backend
              << " shape=" << hdr.hidden << "x" << hdr.inter << " cold_tokens=" << cold << std::endl;
    std::cout << "[CiM " << cim_id << "] latency_ns_per_token=" << lat_ns
              << " energy_nJ_per_token=" << eng_nJ << std::endl;
    std::cout << "[CiM " << cim_id << "] cim_total_latency_ns=" << (lat_ns * (double)cold)
              << " cim_total_energy_nJ=" << total_energy_nJ
              << " cim_compute_cycles=" << compute_cycles << std::endl;

    // 4) Combine: return processed cold-token hidden states + a summary count.
    if (payload_bytes > 0) {
        std::vector<unsigned char> out(payload_bytes, 9);
        cim_send_message(cpu_x, cpu_y, me_x, me_y, out.data(), payload_bytes);
    }
    int32_t completed = cold;
    cim_send_message(cpu_x, cpu_y, me_x, me_y, &completed, sizeof(completed));
    return 0;
}
