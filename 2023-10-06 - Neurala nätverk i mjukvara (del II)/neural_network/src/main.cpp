/********************************************************************************
 * @brief Implementation of neural network in C++. First dense layers will be
 *        implemented. After that, a neural network consisting of multiple dense
 *        layers will be implemented and tested.
 ********************************************************************************/
#include <iostream>
#include <vector>
#include <dense_layer.hpp>

namespace ml = yrgo::machine_learning;

/********************************************************************************
 * @brief Creating a dense layer consisting of two nodes and three weights per
 *        node. Next lecture, functionality will be implemented for 
 *        backpropagation and optimization.
 ********************************************************************************/
int main(void) {
    const std::vector<double> inputs{0, 1, 0};
    const std::vector<double> outputs{1};
    ml::DenseLayer d1{2, 3};

    for (std::size_t i{}; i < 1000; ++i) { 
        d1.Feedforward(inputs); 
        /* Backpropagation. */
        /* Optimization. */
    }
    return 0;
}