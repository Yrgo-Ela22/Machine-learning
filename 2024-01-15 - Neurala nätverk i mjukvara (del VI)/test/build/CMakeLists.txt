################################################################################
# @brief Builds units tests of modules implemented for neural networks.
################################################################################
cmake_minimum_required(VERSION 3.20)
project(neural_network_tests)
find_package(GTest REQUIRED)
include_directories(../../inc ${GTEST_INCLUDE_DIRS})

################################################################################
# @brief Adds executable for testing the DenseLayer class.
################################################################################
add_executable(run_dense_layer_test ../src/dense_layer_test.cpp ../../src/dense_layer.cpp)
target_compile_options(run_dense_layer_test PRIVATE -Wall -Werror)
target_link_libraries(run_dense_layer_test pthread ${GTEST_LIBRARIES})
set_target_properties(run_dense_layer_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../output)