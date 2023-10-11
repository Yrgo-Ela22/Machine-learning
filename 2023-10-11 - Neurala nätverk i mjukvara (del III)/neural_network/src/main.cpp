/********************************************************************************
 * @brief Implementation of neural network in C++. First dense layers will be
 *        implemented. After that, a neural network consisting of multiple dense
 *        layers will be implemented and tested.
 ********************************************************************************/
#include <iostream>
#include <iomanip>
#include <vector>
#include <dense_layer.hpp>

namespace ml = yrgo::machine_learning;

/********************************************************************************
 * @brief Creating a dense layer consisting of two nodes and three weights per
 *        node. The dense layer is trained during 20 epochs, followed by test
 *        of the model. The output of the model matches the reference values
 *        from the training data.
 ********************************************************************************/
int main(void) {
    const std::vector<double> inputs{0, 1, 0};
    const std::vector<double> outputs{0, 1}; 
    ml::DenseLayer d1{2, 3};

    for (std::size_t i{}; i < 100; ++i) { 
        d1.Feedforward(inputs); 
        d1.Backpropagate(outputs);
        d1.Optimize(inputs, 0.1);
    }

    std::cout << "\nOutput values (after optimization):\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    for (auto& i : d1.Output()) {
        std::cout << std::fixed << std::setprecision(2) << i << "\n";
    }
    std::cout << "--------------------------------------------------------------------------------\n\n";
    return 0;
}