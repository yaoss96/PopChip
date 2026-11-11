cmake_minimum_required(VERSION 3.10)
project(All2allMoEE2E LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{SIMULATOR_ROOT})
  get_filename_component(
    SIMULATOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
else()
  set(SIMULATOR_ROOT $ENV{SIMULATOR_ROOT})
endif()

set(INTERCHIPLET_INCLUDE ${SIMULATOR_ROOT}/interchiplet/includes)
set(INTERCHIPLET_C_LIB ${SIMULATOR_ROOT}/interchiplet/lib/libinterchiplet_c.a)

# The in-tree GPGPU-Sim runtime is cuda-11050 and its PTX loader shells out to
# ptxas; CUDA 12's nvcc emits .version 8.0 PTX that the bundled ptxas (7.5)
# rejects, and CUDA 11.5's nvcc refuses host gcc>11. Pin BOTH: use CUDA 11.5
# nvcc with gcc-11 as the host compiler so emitted PTX is <=7.x and the build
# host check passes. Overridable from the environment.
if(DEFINED ENV{MOE_NVCC})
  set(MOE_NVCC $ENV{MOE_NVCC})
else()
  set(MOE_NVCC /usr/local/cuda-11.5/bin/nvcc)
endif()
if(DEFINED ENV{MOE_HOST_CXX})
  set(MOE_HOST_CXX $ENV{MOE_HOST_CXX})
else()
  set(MOE_HOST_CXX g++-11)
endif()

include_directories(${INTERCHIPLET_INCLUDE})

add_executable(moe_router_cpu moe_router_cpu.cpp)
target_link_libraries(moe_router_cpu ${INTERCHIPLET_C_LIB} pthread)

# Sniper's bundled Pin (pin-3.11) cannot parse modern Ubuntu glibc/loader
# ELF sections such as .relr.dyn. Build a static router binary for the
# detailed CPU path so run-sniper does not inject through the host loader.
add_executable(moe_router_cpu_static moe_router_cpu.cpp)
target_link_options(moe_router_cpu_static PRIVATE -static)
target_link_libraries(moe_router_cpu_static ${INTERCHIPLET_C_LIB} pthread)

# Dedicated CiM expert chiplet (P1): plain C++, same InterChiplet lib as the router.
add_executable(moe_cim_worker moe_cim_worker.cpp)
target_link_libraries(moe_cim_worker ${INTERCHIPLET_C_LIB} pthread)
add_custom_command(TARGET moe_cim_worker POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moe_cim_worker> ${CMAKE_BINARY_DIR}/bin/moe_cim_worker
)

# Plain-C++ GPU stub (no CUDA) -- lets the full hetero co-sim run without GPGPU-Sim.
add_executable(moe_gpu_stub moe_gpu_stub.cpp)
target_link_libraries(moe_gpu_stub ${INTERCHIPLET_C_LIB} pthread)
add_custom_command(TARGET moe_gpu_stub POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moe_gpu_stub> ${CMAKE_BINARY_DIR}/bin/moe_gpu_stub
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/cuobj ${CMAKE_BINARY_DIR}/bin)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/cuobj/apis_cu.o
  COMMAND ${MOE_NVCC} -ccbin ${MOE_HOST_CXX} --compiler-options -Wall -I${INTERCHIPLET_INCLUDE} -c ${CMAKE_CURRENT_SOURCE_DIR}/apis_cu.cpp -o ${CMAKE_BINARY_DIR}/cuobj/apis_cu.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/apis_cu.cpp
)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_worker.o
  COMMAND ${MOE_NVCC} -ccbin ${MOE_HOST_CXX} -DGPGPUSIM_LEGACY_LAUNCH --compiler-options -Wall -I${INTERCHIPLET_INCLUDE} -c ${CMAKE_CURRENT_SOURCE_DIR}/moe_gpu_worker.cu -o ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_worker.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/moe_gpu_worker.cu
)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/bin/moe_gpu_worker
  COMMAND ${MOE_NVCC} -ccbin ${MOE_HOST_CXX} -L${SIMULATOR_ROOT}/gpgpu-sim/lib/$ENV{GPGPUSIM_CONFIG} --cudart shared ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_worker.o ${CMAKE_BINARY_DIR}/cuobj/apis_cu.o ${INTERCHIPLET_C_LIB} -lpthread -o ${CMAKE_BINARY_DIR}/bin/moe_gpu_worker
  DEPENDS ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_worker.o ${CMAKE_BINARY_DIR}/cuobj/apis_cu.o ${INTERCHIPLET_C_LIB}
)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_kernel_runner.o
  COMMAND ${MOE_NVCC} -ccbin ${MOE_HOST_CXX} -DGPGPUSIM_LEGACY_LAUNCH --compiler-options -Wall -c ${CMAKE_CURRENT_SOURCE_DIR}/moe_gpu_kernel_runner.cu -o ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_kernel_runner.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/moe_gpu_kernel_runner.cu
)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_kernel_runner_shim.o
  COMMAND ${MOE_NVCC} -ccbin ${MOE_HOST_CXX} --compiler-options -Wall -c ${CMAKE_CURRENT_SOURCE_DIR}/moe_gpu_kernel_runner_shim.cpp -o ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_kernel_runner_shim.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/moe_gpu_kernel_runner_shim.cpp
)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/bin/moe_gpu_kernel_runner
  COMMAND ${MOE_NVCC} -ccbin ${MOE_HOST_CXX} -L${SIMULATOR_ROOT}/gpgpu-sim/lib/$ENV{GPGPUSIM_CONFIG} --cudart shared ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_kernel_runner.o ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_kernel_runner_shim.o -o ${CMAKE_BINARY_DIR}/bin/moe_gpu_kernel_runner
  DEPENDS ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_kernel_runner.o ${CMAKE_BINARY_DIR}/cuobj/moe_gpu_kernel_runner_shim.o
)

add_custom_target(moe_gpu_worker ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/moe_gpu_worker)
add_custom_target(moe_gpu_kernel_runner ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/moe_gpu_kernel_runner)

add_custom_command(TARGET moe_router_cpu POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moe_router_cpu> ${CMAKE_BINARY_DIR}/bin/moe_router_cpu
)

add_custom_command(TARGET moe_router_cpu_static POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moe_router_cpu_static> ${CMAKE_BINARY_DIR}/bin/moe_router_cpu_static
)
