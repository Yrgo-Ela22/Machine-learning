/********************************************************************************
 * @brief Implementation of dense layers in C++. Later on, this dense layer
 *        implementation will be used to create conventional neural networks.
 ********************************************************************************/
#include <iostream>
#include <iomanip>
#include <vector>
#include <dense_layer.hpp>

using namespace yrgo::machine_learning;

namespace {

// --------------------------------------------------------------------------------
void Train(DenseLayer& layer, 
           const std::vector<double>& inputs, 
           const std::vector<double>& outputs,
           const std::size_t num_epochs,
           const double learning_rate = 0.01) {
    for (std::size_t i{}; i < num_epochs; ++i) { 
        layer.Feedforward(inputs); 
        layer.Backpropagate(outputs);
        layer.Optimize(inputs, learning_rate);
    }
}

// --------------------------------------------------------------------------------
void Print(const DenseLayer& layer, 
           const std::size_t num_decimals = 0,
           std::ostream& ostream = std::cout) {
    ostream << std::fixed << std::setprecision(num_decimals);
    ostream << "--------------------------------------------------------------------------------\n";
    for (auto& i : layer.Output()) {
        ostream << i << "\n";
    }
    ostream << "--------------------------------------------------------------------------------\n\n";
}

} /* namespace */

/********************************************************************************
 * @brief Creating a dense layer consisting of two nodes and three weights per
 *        node. The dense layer is trained during 100 epochs, followed by test
 *        of the model. The output of the model matches the reference values
 *        from the training data.
 ********************************************************************************/
int main(void) {
    const std::vector<double> inputs{0, 1, 0};
    const std::vector<double> outputs{0, 1}; 
    DenseLayer layer{outputs.size(), inputs.size()};
    Train(layer, inputs, outputs, 100, 0.1);
    Print(layer, 1);
    return 0;
}