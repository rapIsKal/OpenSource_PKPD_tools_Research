project(benchmark CXX)
cmake_minimum_required(VERSION 2.6.2)

add_library(benchmark STATIC Benchmark.cpp)
target_include_directories(benchmark PUBLIC ${benchmark_SOURCE_DIR})

add_library(benchmark_main STATIC benchmark_main.cpp)
target_include_directories(benchmark_main PUBLIC ${benchmark_SOURCE_DIR})
