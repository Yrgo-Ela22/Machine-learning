/********************************************************************************
 * @brief Implementation of simple neural network in C++
 ********************************************************************************/
#include <vector>

#include <neural_network.hpp>

using namespace yrgo::machine_learning;

/********************************************************************************
 * @brief Creates a neural network trained to predict a 2-bit XOR pattern.
 *        The network consists of two inputs, two hidden nodes and one output.
 *        TanH is used as activation function in the hidden layer in order
 *        to make the network better at predicting complex patterns, while
 *        ReLU is used in the output layer.
 * 
 *        The model is trained during 1000 epochs with a 10 % learning rate.
 *        If the training is successful, the training inputs are used for
 *        prediction, which is printed in the terminal.
 ********************************************************************************/
int main(void) {
    /* Comments out the code until the implementation is finished.: */
    #if 0 
    const std::vector<std::vector<double>> train_input{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> train_output{{0}, {1}, {1}, {0}};

    NeuralNetwork network{2, 2, 1, ActFunc::kTanh};
    network.AddTrainingData(train_input, train_output);
    if (network.Train(1000, 0.1)) {
        network.PrintPredictions(train_input);
    }
    #endif
    return 0;
}