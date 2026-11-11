// Standalone fail-closed parser regression test; no simulator libraries needed.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "moe_coords.h"

struct ParserCase {
    const char *name;
    const char *text;
    bool valid;
};

int main() {
    const std::vector<ParserCase> cases = {
        {"valid",
         "# order-free, inline-comment input\n"
         "gpu 1 2 0 # second worker\n"
         "width 4\nrouter 3 1\nngpu 2\ngpu 0 0 0\n",
         true},
        {"duplicate_key",
         "width 4\nwidth 4\nrouter 3 1\nngpu 1\ngpu 0 0 0\n",
         false},
        {"duplicate_id",
         "width 4\nrouter 3 1\nngpu 1\ngpu 0 0 0\ngpu 0 1 0\n",
         false},
        {"missing_id",
         "width 4\nrouter 3 1\nngpu 3\ngpu 0 0 0\ngpu 2 2 0\n",
         false},
        {"count_mismatch",
         "width 4\nrouter 3 1\nngpu 2\ngpu 0 0 0\ngpu 1 1 0\ngpu 2 2 0\n",
         false},
        {"negative_coordinate",
         "width 4\nrouter 3 1\nngpu 1\ngpu 0 -1 0\n",
         false},
        {"x_out_of_width",
         "width 4\nrouter 3 1\nngpu 1\ngpu 0 4 0\n",
         false},
        {"duplicate_coordinate",
         "width 4\nrouter 3 1\nngpu 2\ngpu 0 0 0\ngpu 1 0 0\n",
         false},
        {"router_conflict",
         "width 4\nrouter 3 1\nngpu 1\ngpu 0 3 1\n",
         false},
        {"unknown_key",
         "width 4\nrouter 3 1\nngpu 1\ngpu 0 0 0\nextra 7\n",
         false},
    };

    int failures = 0;
    std::string valid_path;
    std::string invalid_path;
    for (const auto &test : cases) {
        const std::string path = "/tmp/ilp4moe_coords_" +
            std::to_string(static_cast<long long>(getpid())) + "_" +
            test.name + ".txt";
        {
            std::ofstream out(path);
            out << test.text;
        }
        const moe::CoordMap parsed = moe::load_coord_file(path);
        if (!parsed.configured || parsed.valid != test.valid) {
            std::cerr << "FAIL case=" << test.name
                      << " configured=" << parsed.configured
                      << " valid=" << parsed.valid << std::endl;
            ++failures;
        }
        if (test.valid) {
            valid_path = path;
            if (parsed.width != 4 || parsed.ngpu != 2 ||
                parsed.gpu.size() != 2) {
                std::cerr << "FAIL valid payload fields" << std::endl;
                ++failures;
            }
        } else if (invalid_path.empty()) {
            invalid_path = path;
        }
    }

    unsetenv("MOE_COORD_FILE");
    const moe::CoordMap legacy = moe::load_coord_env();
    if (legacy.configured || legacy.valid) {
        std::cerr << "FAIL unset environment must select legacy state"
                  << std::endl;
        ++failures;
    }
    setenv("MOE_COORD_FILE", invalid_path.c_str(), 1);
    const moe::CoordMap formal_invalid = moe::load_coord_env();
    if (!formal_invalid.configured || formal_invalid.valid) {
        std::cerr << "FAIL configured invalid environment was not preserved"
                  << std::endl;
        ++failures;
    }
    setenv("MOE_COORD_FILE", valid_path.c_str(), 1);
    const moe::CoordMap formal_valid = moe::load_coord_env();
    if (!formal_valid.configured || !formal_valid.valid) {
        std::cerr << "FAIL configured valid environment was rejected"
                  << std::endl;
        ++failures;
    }
    unsetenv("MOE_COORD_FILE");

    for (const auto &test : cases) {
        const std::string path = "/tmp/ilp4moe_coords_" +
            std::to_string(static_cast<long long>(getpid())) + "_" +
            test.name + ".txt";
        std::remove(path.c_str());
    }
    if (failures) return 1;
    std::cout << "MOE_COORDS_PARSER_TEST=PASS cases=" << cases.size()
              << std::endl;
    return 0;
}
