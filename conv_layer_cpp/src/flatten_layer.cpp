/********************************************************************************
 * @brief Implementation details of the ml::FlattenLayer class.
 ********************************************************************************/
#include "conv_utils.h"
#include "flatten_layer.h"

namespace ml
{

// -----------------------------------------------------------------------------
FlattenLayer::FlattenLayer() = default;

// -----------------------------------------------------------------------------
FlattenLayer::FlattenLayer(const std::vector<std::vector<double>>& input)
{
    feedforward(input);
}

// -----------------------------------------------------------------------------
const std::vector<double>& FlattenLayer::output() const { return myOutput; }

// -----------------------------------------------------------------------------
const std::vector<double>& FlattenLayer::error() const { return myError; }

// -----------------------------------------------------------------------------
void FlattenLayer::feedforward(const std::vector<std::vector<double>>& input)
{
    myOutput.clear();
    for (const auto& i : input)
    {
        for (const auto& j : i)
        {
            myOutput.push_back(j);
        }
    }
}

// -----------------------------------------------------------------------------
void FlattenLayer::backpropagate(const std::vector<double>& nextLayerError)
{
    myError = nextLayerError;
}

} // namespace ml