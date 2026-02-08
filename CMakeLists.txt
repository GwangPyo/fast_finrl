cmake_minimum_required(VERSION 3.22)
project(fast_finrl)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include(FetchContent)

# FetchContent - system package 없는 라이브러리
FetchContent_Declare(
    DataFrame
    GIT_REPOSITORY https://github.com/hosseinmoein/DataFrame.git
    GIT_TAG master
)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG 0.7.7
)
FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG 0.25.0
)
FetchContent_MakeAvailable(DataFrame pybind11 xtl xtensor)

# System packages
find_package(nlohmann_json REQUIRED)
find_package(TBB REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# Library
add_library(fast_finrl_lib
    src/FastFinRL.cpp
    src/DataLoader.cpp
    src/StateSerializer.cpp
    src/ReplayBuffer.cpp
    src/VecFastFinRL.cpp
    src/VecReplayBuffer.cpp
)
target_include_directories(fast_finrl_lib PUBLIC include)
target_link_libraries(fast_finrl_lib PUBLIC
    DataFrame::DataFrame
    nlohmann_json::nlohmann_json
    TBB::tbb
    Arrow::arrow_shared
    Parquet::parquet_shared
    xtensor
)

# Python bindings
pybind11_add_module(fast_finrl_py src/bindings.cpp)
target_link_libraries(fast_finrl_py PRIVATE fast_finrl_lib)

install(TARGETS fast_finrl_py LIBRARY DESTINATION .)
