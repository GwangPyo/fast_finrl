cmake_minimum_required(VERSION 3.22)
project(fast_finrl)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Release build with O3 optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# OpenMP disabled - DataFrame library has internal #pragma omp simd in lambdas that conflict
# find_package(OpenMP REQUIRED)

# FetchContent for dependencies
include(FetchContent)

# fmt library (for <format> polyfill on older compilers)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
)

# hosseinmoein/DataFrame
FetchContent_Declare(
    DataFrame
    GIT_REPOSITORY https://github.com/hosseinmoein/DataFrame.git
    GIT_TAG master
)

# nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)

FetchContent_MakeAvailable(DataFrame json googletest pybind11)

# Main library
add_library(fast_finrl_lib
    src/FastFinRL.cpp
)
target_include_directories(fast_finrl_lib PUBLIC include)
set_target_properties(fast_finrl_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(TBB REQUIRED)
target_link_libraries(fast_finrl_lib PUBLIC DataFrame::DataFrame nlohmann_json::nlohmann_json TBB::tbb)

# Main executable
add_executable(fast_finrl main.cpp)
target_link_libraries(fast_finrl PRIVATE fast_finrl_lib)

# Tests
enable_testing()
add_executable(fast_finrl_tests tests/test_fast_finrl.cpp)
target_link_libraries(fast_finrl_tests PRIVATE fast_finrl_lib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(fast_finrl_tests)

# Python bindings
pybind11_add_module(fast_finrl_py src/bindings.cpp)
target_link_libraries(fast_finrl_py PRIVATE fast_finrl_lib)

# Install Python module (for pip install)
install(TARGETS fast_finrl_py LIBRARY DESTINATION .)
