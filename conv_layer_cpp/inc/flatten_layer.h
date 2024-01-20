/********************************************************************************
 * @brief Implementation of flatten layers for conversion of two-dimensional 
 *        vectors to one dimension. The one-dimensional output can be used
 *        as input on a conventional neural network.
 ********************************************************************************/
#pragma once

#include <vector>

namespace ml
{

/********************************************************************************
 * @brief Class for implementation of flatten layers. The size of the vectors 
 *        to flatten is dynamic.
 ********************************************************************************/
class FlattenLayer
{
public:

    /********************************************************************************
     * @brief Creates new flatten layer.
     ********************************************************************************/
    FlattenLayer();

    /********************************************************************************
     * @brief Creates new flatten layer and flattens referenced input.
     * 
     * @param input Reference to vector holding the input data to flatten.
     ********************************************************************************/
    FlattenLayer(const std::vector<std::vector<double>>& input);

    /********************************************************************************
     * @brief Provides the one-dimensional output of the flatten layer.
     * 
     * @return Reference to vector holding the output of the flatten layer.
     ********************************************************************************/
    const std::vector<double>& output() const;

    /********************************************************************************
     * @brief Provides the error values from the next layer (which should be a
     *        dense layer).
     * 
     * @return Reference to vector holding the error values from next layer. 
     ********************************************************************************/
    const std::vector<double>& error() const;

    /********************************************************************************
     * @brief Flattens referenced input.
     * 
     * @param input Reference to vector holding the input data to flatten.
     ********************************************************************************/
    void feedforward(const std::vector<std::vector<double>>& input);

    /********************************************************************************
     * @brief Stored error values from next layer (which should be a dense layer).
     * 
     * @param nextLayerError Reference to vector holding error values from next layer.
     ********************************************************************************/
    void backpropagate(const std::vector<double>& nextLayerError);

protected:
    std::vector<double> myOutput{};
    std::vector<double> myError{};
};

} // namespace ml