/********************************************************************************
 * @brief Demonstration of a two-dimensional convolutional layer trained with
 *        a 5 x 5 image. The output of the convolutional layer is flattened to
 *        one dimension via a flatten layer. The flattened input could be used 
 *        as input to a sequential dense layer in a neural network. 
 ********************************************************************************/
#include <iostream>
#include <vector>

#include "conv_layer_2d.h"
#include "conv_utils.h"
#include "flatten_layer.h"

using namespace ml::utils;

/********************************************************************************
 * @brief Creates a two-dimensional convolutional layer with kernel size 3 x 3.
 *        The convolutional layer is fed with a 5 x 5 image. Kernel and input
 *        error values are calculated with error values from an arbitrary next
 *        layer. The kernel parameters are then modified via optimization with 
 *        a 1 % learning rate. The output of the convolutional layer is flattened 
 *        to one dimension via a flatten layer.
 * 
 *        The parameters of the convolutional layer are printed, along with
 *        the output of the flatten layer, before terminating the program.
 * 
 * @return Success code 0 upon termination of the program.
 ********************************************************************************/
int main()
{
    ml::ConvLayer2D convLayer{3};
    const std::vector<std::vector<double>> input(5, std::vector<double>{1, 2, 3, 4, 5});
    const std::vector<std::vector<double>> outputError(
        input.size(), std::vector<double>(input.size(), 1));
    std::cout << "\nKernel before optimization:\n";
    print(convLayer.kernel(), 1);

    convLayer.feedforward(input);
    convLayer.backpropagate(outputError);
    convLayer.optimize(0.01);
    ml::FlattenLayer flattenLayer{convLayer.output()};

    std::cout << "Padded input:\n";
    print(convLayer.inputPadded());
    std::cout << "Kernel after optimization:\n";
    print(convLayer.kernel(), 1);
    std::cout << "Output:\n";
    print(convLayer.output());
    std::cout << "Flattened output:\n";
    print(flattenLayer.output());
    return 0;
}