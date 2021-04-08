cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++2a")
project(kmeans)

add_executable(
	kmeans
	src/main.cpp
)

target_link_libraries(kmeans PRIVATE fmt)
target_link_libraries(kmeans PRIVATE mlpack)
target_link_libraries(kmeans PRIVATE armadillo)
