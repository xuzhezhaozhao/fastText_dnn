project(fasttext)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -O3 -funroll-loops -pthread")

include_directories(./deps/lshbox)
include_directories(./deps/cppjieba/include)
include_directories(./deps/cppjieba/deps/)
include_directories(./)


link_directories(${CMAKE_SOURCE_DIR}/deps/openblas/lib)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(utils)
add_subdirectory(deps)
