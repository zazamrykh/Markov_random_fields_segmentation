cmake_minimum_required(VERSION 3.25)
project(markov_segmentation)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(markov_segmentation main.cpp)

target_link_libraries(markov_segmentation ${OpenCV_LIBS})