/********************************************************************************
 * @brief Implementation of simple neural network in C++
 ********************************************************************************/
#include <vector>

#include <neural_network.hpp>

using namespace yrgo::machine_learning;

/********************************************************************************
 * @brief Creates a neural network trained to predict a 2-bit XOR pattern.
 *        The network consists of two inputs, three hidden nodes and one output.
 *        TanH is used as activation function in the hidden layer in order
 *        to make the network better at predicting complex patterns, while
 *        ReLU is used in the output layer.
 * 
 *        The model is trained during 10 000 epochs with a 1 % learning rate.
 *        If the training is successful, the training inputs are used for
 *        prediction, which is printed in the terminal.
 ********************************************************************************/
int main(void) {
    const std::vector<std::vector<double>> train_input{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> train_output{{0}, {1}, {1}, {0}};

    NeuralNetwork network{2, 3, 1, ActFunc::kTanh};
    network.AddTrainingData(train_input, train_output);
    if (network.Train(10000, 0.01)) {
        network.PrintPredictions(train_input);
    }
    return 0;
}