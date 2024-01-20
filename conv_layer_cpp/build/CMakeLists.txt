cmake_minimum_required(VERSION 3.20)
project(conv_layer_1d)

set(EXECUTABLE "${CMAKE_PROJECT_NAME}")

include_directories(../inc)
add_executable(${EXECUTABLE} ../src/conv_layer_1d.cpp
                             ../src/conv_layer_2d.cpp
                             ../src/flatten_layer.cpp 
                             ../src/main.cpp)
target_compile_options(${EXECUTABLE} PRIVATE -Wall -Werror)
set_target_properties(${EXECUTABLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../out)
set_property(TARGET ${EXECUTABLE} PROPERTY CXX_STANDARD 17)