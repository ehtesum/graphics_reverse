cmake_minimum_required(VERSION 3.20)
project(GTX1650_Zen2_Research LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments
)

# Optional CUDA support if nvcc is available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    file(GLOB EXPERIMENT_SOURCES "experiments/*.cu")
    foreach(EXP_SRC ${EXPERIMENT_SOURCES})
        get_filename_component(EXP_NAME ${EXP_SRC} NAME_WE)
        add_executable(${EXP_NAME} ${EXP_SRC})
        set_target_properties(${EXP_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
        )
    endforeach()
endif()

# CPU microbenchmarks
file(GLOB CPU_BENCH_SOURCES "cpu/benchmarks/*.cpp")
foreach(CPU_SRC ${CPU_BENCH_SOURCES})
    get_filename_component(BENCH_NAME ${CPU_SRC} NAME_WE)
    add_executable(${BENCH_NAME} ${CPU_SRC})
    if(MSVC)
        target_compile_options(${BENCH_NAME} PRIVATE /arch:AVX2 /O2)
    else()
        target_compile_options(${BENCH_NAME} PRIVATE -mavx2 -O3 -pthread)
    endif()
endforeach()
