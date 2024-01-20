/********************************************************************************
 * @brief Implementation details of the ml::ConvLayer2D class.
 ********************************************************************************/
#include "conv_layer_2d.h"
#include "conv_utils.h"

namespace ml
{

// -----------------------------------------------------------------------------
ConvLayer2D::ConvLayer2D(const std::size_t kernelSize)
{
    initKernel(kernelSize);
}

// -----------------------------------------------------------------------------
const std::vector<std::vector<double>>& ConvLayer2D::inputPadded() const
{
    return myInputPadded;
}

// -----------------------------------------------------------------------------
const std::vector<std::vector<double>>& ConvLayer2D::kernel() const
{
    return myKernel;
}

// -----------------------------------------------------------------------------
const std::vector<std::vector<double>>& ConvLayer2D::output() const
{
    return myOutput;
}

// -----------------------------------------------------------------------------
const std::vector<std::vector<double>>& ConvLayer2D::inputError() const
{
    return myInputError;
}

// -----------------------------------------------------------------------------
const std::vector<std::vector<double>>& ConvLayer2D::kernelError() const
{
    return myKernelError;
}

// -----------------------------------------------------------------------------
std::size_t ConvLayer2D::imageWidth() const { return myOutput.size(); }

// -----------------------------------------------------------------------------
std::size_t ConvLayer2D::imageHeight() const
{
    return myOutput.size() > 0 ? myOutput[0].size() : 0;
}

// -----------------------------------------------------------------------------
std::size_t ConvLayer2D::kernelSize() const { return myKernel.size(); }

// -----------------------------------------------------------------------------
void ConvLayer2D::feedforward(const std::vector<std::vector<double>>& input)
{
    if (input.empty()) { return; }
    myInputPadded = pad(input);
    myOutput.resize(input.size(), std::vector<double>(input[0].size(), 0));

    for (std::size_t i{}; i < imageWidth(); ++i)
    {
        for (std::size_t j{}; j < imageHeight(); ++j)
        {
            for (std::size_t k{}; k < kernelSize(); ++k)
            {
                for (std::size_t l{}; l < kernelSize(); ++l)
                {
                    myOutput[i][j] += myInputPadded[i + k][j + l] * myKernel[k][l];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
void ConvLayer2D::backpropagate(const std::vector<std::vector<double>>& outputError)
{
    myKernelError.resize(kernelSize(), std::vector<double>(kernelSize(), 0));
    myInputError.resize(imageWidth(), std::vector<double>(imageHeight(), 0));
    const std::vector<std::vector<double>> outputErrorPadded{pad(outputError)};
    const auto offset{kernelSize() - 1};

    for (std::size_t i{}; i < imageWidth(); ++i)
    {
        for (std::size_t j{}; j < imageHeight(); ++j)
        {
            for (std::size_t k{}; k < kernelSize(); ++k)
            {
                for (std::size_t l{}; l < kernelSize(); ++l)
                {
                    myKernelError[k][l] += myInputPadded[i + k][j + l] * outputError[i][j];
                    myInputError[i][j] += 
                        outputErrorPadded[offset + i - k][offset + j - l] * myKernel[k][l];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
void ConvLayer2D::optimize(const double learningRate)
{
    for (std::size_t i{}; i < kernelSize(); ++i)
    {
        for (std::size_t j{}; j < kernelSize(); ++j)
        {
            myKernel[i][j] += myKernelError[i][j] * learningRate;
        }
    }
}

// -----------------------------------------------------------------------------
void ConvLayer2D::initKernel(const std::size_t kernelSize)
{
    myKernel.resize(kernelSize, std::vector<double>(kernelSize));
    for (auto& i : myKernel)
    {
        for (auto& j : i)
        {
            j = utils::random<double>(0, 1);
        }
    }
}

// -----------------------------------------------------------------------------
std::vector<std::vector<double>> ConvLayer2D::pad(const std::vector<std::vector<double>>& input)
{
    return utils::pad<double>(input, myKernel);
}

// -----------------------------------------------------------------------------
std::size_t ConvLayer2D::numPaddings() const
{
    return kernelSize() / 2;
}

// -----------------------------------------------------------------------------
std::size_t ConvLayer2D::width(const std::vector<std::vector<double>>& input)
{
    return input.size();
}

// -----------------------------------------------------------------------------
std::size_t ConvLayer2D::height(const std::vector<std::vector<double>>& input)
{
    return input.size() > 0 ? input[0].size() : 0;
}

} // namespace ml