/********************************************************************************
 * @brief Unit tests for dense layers consisting of two nodes and three weights
 *        per node. The dense layers are trained to predict the number of high
 *        inputs (0 - 3) in binary form (0b00 - 0b11). The dense layers are
 *        trained during 1000 epochs each with a 1 % learning rate. 
 ********************************************************************************/
#include <gtest/gtest.h>
#include <vector>
#include <dense_layer.hpp>

using namespace yrgo::machine_learning;

namespace {

void TrainLayer(DenseLayer& layer,
                const std::vector<double>& input, 
                const std::vector<double>& output, 
                const std::size_t num_epochs = 1000, 
                const double learning_rate = 0.01) {
    for (std::size_t i{}; i < num_epochs; ++i) {
        layer.Feedforward(input);
        layer.Backpropagate(output);
        layer.Optimize(input, learning_rate);
    }
}

void CheckResult(const DenseLayer& layer,  
                 const std::vector<double>& output) {
    for (std::size_t i{}; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], layer.Output()[i], 0.001);
    }
}

void RunTest(const std::vector<double>& input, 
             const std::vector<double>& output, 
             const std::size_t num_epochs = 1000, 
             const double learning_rate = 0.01) {
    DenseLayer layer{output.size(), input.size()};
    TrainLayer(layer, input, output, num_epochs, learning_rate);
    CheckResult(layer, output);
}

void RunTests(const std::vector<std::vector<double>>& inputs, 
              const std::vector<std::vector<double>>& outputs, 
              const std::size_t num_epochs = 1000, 
              const double learning_rate = 0.01) {
    for (std::size_t i{}; i < inputs.size() && i < outputs.size(); ++i) {
        RunTest(inputs[i], outputs[i], num_epochs, learning_rate);
    }
}

TEST(DenseLayerTest, AllTests) {
    const std::vector<std::vector<double>> inputs{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                                  {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    const std::vector<std::vector<double>> outputs{{0, 0}, {0, 1}, {0, 1}, {1, 0},
                                                   {0, 1}, {1, 0}, {1, 0}, {1, 1}};
    RunTests(inputs, outputs);
}

} /* namespace */

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}