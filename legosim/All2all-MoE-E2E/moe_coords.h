// moe_coords.h -- shared layout-coordinate map for the MoE E2E benchmark.
//
// Task 3 of the ILP4MOE closed loop: the stock router/worker binaries hard-code
// the chiplet coordinates ({0,0}..{0,3}) and cap the dispatch matrix at 4x4,
// so an ILP-optimized 3D layout (arbitrary (x,y) sites, up to |V| GPU experts)
// could never actually drive the NoC. This header lets BOTH moe_router_cpu and
// moe_gpu_worker read a common coordinate file emitted by src/codegen.py.
//
// Contract with codegen.py (Option-A coordinate flattening, design doc Sec 6/7):
//   (l,i,j) -> row=i, col=l*Y+j, node = row*(L*Y)+col
// InterChiplet linearizes a NoC node as  node = x + y*width  (net_bench.h
// dumpBench, width == interchiplet's -w flag). Equating the two requires
//   x = col,  y = row,  width = C = L*Y
// so the coordinate file records width=C and one (x,y)=(col,row) per site.
//
// File format (text, '#'-comments and blank lines ignored, order-free keys):
//   width  <W>            popnet mesh width == interchiplet -w
//   router <X> <Y>        CPU/router chiplet coordinate (x,y)
//   ngpu   <N>            number of GPU expert chiplets
//   gpu    <ID> <X> <Y>   one line per GPU worker, ID in [0,N)
//
// ACTIVATION: when MOE_COORD_FILE is unset, callers retain the legacy 4-GPU
// smoke coordinates.  Once it is set, however, parsing is formal/fail-closed:
// an unreadable or malformed file must terminate the caller and must never
// silently select the legacy coordinates.

#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace moe {

struct Coord {
    int x = 0;
    int y = 0;
};

struct CoordMap {
    int width = 0;                 // popnet mesh width (== interchiplet -w)
    Coord router;                  // CPU/router node
    int ngpu = 0;                  // number of GPU expert chiplets
    std::vector<Coord> gpu;        // gpu[id] = (x,y)
    bool configured = false;       // a coordinate file was explicitly requested
    bool valid = false;
};

// Parse a coordinate file. Returns a CoordMap with .valid=true on success.
// On any structural error it logs to stderr and returns configured=true,
// valid=false; formal callers must reject that state.
inline CoordMap load_coord_file(const std::string &path) {
    CoordMap cm;
    cm.configured = true;
    auto fail = [&](const std::string &message) {
        std::cerr << "[moe_coords] ERROR: " << message << " in " << path
                  << std::endl;
        return cm;
    };
    std::ifstream fin(path);
    if (!fin.is_open()) {
        return fail("cannot open coordinate file");
    }
    bool have_width = false;
    bool have_router = false;
    bool have_ngpu = false;
    std::vector<Coord> gpu_tmp;
    std::vector<bool> gpu_seen;
    int declared_ngpu = -1;
    std::string line;
    int line_number = 0;
    while (std::getline(fin, line)) {
        ++line_number;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        std::size_t b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        std::istringstream iss(line.substr(b));
        std::string key;
        iss >> key;
        if (key == "width") {
            if (have_width) return fail("duplicate 'width' key");
            std::string extra;
            if (!(iss >> cm.width) || (iss >> extra))
                return fail("malformed 'width' record at line " +
                            std::to_string(line_number));
            have_width = true;
        } else if (key == "router") {
            if (have_router) return fail("duplicate 'router' key");
            std::string extra;
            if (!(iss >> cm.router.x >> cm.router.y) || (iss >> extra))
                return fail("malformed 'router' record at line " +
                            std::to_string(line_number));
            have_router = true;
        } else if (key == "ngpu") {
            if (have_ngpu) return fail("duplicate 'ngpu' key");
            std::string extra;
            if (!(iss >> declared_ngpu) || (iss >> extra))
                return fail("malformed 'ngpu' record at line " +
                            std::to_string(line_number));
            have_ngpu = true;
        } else if (key == "gpu") {
            int id, x, y;
            std::string extra;
            if (!(iss >> id >> x >> y) || (iss >> extra))
                return fail("malformed 'gpu' record at line " +
                            std::to_string(line_number));
            if (id < 0 || id >= 128)
                return fail("GPU id outside [0,128) at line " +
                            std::to_string(line_number));
            if ((int)gpu_tmp.size() <= id) {
                gpu_tmp.resize(id + 1);
                gpu_seen.resize(id + 1, false);
            }
            if (gpu_seen[id])
                return fail("duplicate GPU id " + std::to_string(id));
            gpu_tmp[id] = Coord{x, y};
            gpu_seen[id] = true;
        } else {
            return fail("unknown key '" + key + "' at line " +
                        std::to_string(line_number));
        }
    }
    if (!have_width || cm.width <= 0)
        return fail("missing/invalid 'width'");
    if (!have_router) return fail("missing 'router'");
    if (!have_ngpu || declared_ngpu <= 0 || declared_ngpu > 128)
        return fail("missing/invalid 'ngpu'");
    if (cm.router.x < 0 || cm.router.y < 0 || cm.router.x >= cm.width)
        return fail("negative/out-of-width router coordinate");
    if ((int)gpu_tmp.size() != declared_ngpu ||
        (int)gpu_seen.size() != declared_ngpu)
        return fail("declared ngpu does not match GPU ID range");
    for (int id = 0; id < declared_ngpu; ++id) {
        if (!gpu_seen[id])
            return fail("missing GPU id " + std::to_string(id));
        if (gpu_tmp[id].x < 0 || gpu_tmp[id].y < 0 ||
            gpu_tmp[id].x >= cm.width)
            return fail("negative/out-of-width coordinate for GPU id " +
                        std::to_string(id));
    }
    std::set<std::pair<int, int>> occupied;
    occupied.insert({cm.router.x, cm.router.y});
    for (int id = 0; id < declared_ngpu; ++id) {
        if (!occupied.insert({gpu_tmp[id].x, gpu_tmp[id].y}).second)
            return fail("duplicate/router-conflicting coordinate for GPU id " +
                        std::to_string(id));
    }
    cm.gpu = gpu_tmp;
    cm.ngpu = declared_ngpu;
    cm.valid = true;
    return cm;
}

// Convenience: configured=false means the legacy smoke fallback was requested
// by leaving MOE_COORD_FILE unset. configured=true,valid=false is a hard error.
inline CoordMap load_coord_env() {
    const char *p = std::getenv("MOE_COORD_FILE");
    if (!p || !*p) return CoordMap{};
    return load_coord_file(std::string(p));
}

}  // namespace moe
