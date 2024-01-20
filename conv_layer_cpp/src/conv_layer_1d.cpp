/********************************************************************************
 * @brief Implementation details of the ml::ConvLayer1D class.
 ********************************************************************************/
#include "conv_layer_1d.h"
#include "conv_utils.h"

namespace ml
{

// -----------------------------------------------------------------------------
ConvLayer1D::ConvLayer1D(const std::size_t kernelSize)
{
    utils::initRandomGenerator();
    initKernel(kernelSize);
}

// -----------------------------------------------------------------------------
const std::vector<double>& ConvLayer1D::inputPadded() const { return myInputPadded; }

// -----------------------------------------------------------------------------
const std::vector<double>& ConvLayer1D::kernel() const { return myKernel; }

// -----------------------------------------------------------------------------
const std::vector<double>& ConvLayer1D::output() const { return myOutput; }

// -----------------------------------------------------------------------------
const std::vector<double>& ConvLayer1D::kernelError() const { return myKernelError; }

// -----------------------------------------------------------------------------
const std::vector<double>& ConvLayer1D::inputError() const { return myInputError; }

// -----------------------------------------------------------------------------
std::size_t ConvLayer1D::imageSize() const { return myOutput.size(); }

// -----------------------------------------------------------------------------
std::size_t ConvLayer1D::kernelSize() const { return myKernel.size(); }

// -----------------------------------------------------------------------------
void ConvLayer1D::feedforward(const std::vector<double>& input)
{
    myInputPadded = pad(input);
    myOutput.resize(input.size(), 0);
    
    for (std::size_t i{}; i < imageSize(); ++i)
    {
        for (std::size_t j{}; j < kernelSize(); ++j)
        {
            myOutput[i] += myInputPadded[i + j] * myKernel[j];
        }
    }
}

// -----------------------------------------------------------------------------
void ConvLayer1D::backpropagate(const std::vector<double>& outputError)
{
    std::vector<double> outputErrorPadded{pad(outputError)};

    myKernelError.resize(kernelSize(), 0);
    myInputError.resize(imageSize(), 0);

    for (std::size_t i{}; i < imageSize(); ++i)
    {
        for (std::size_t j{}; j < kernelSize(); ++j)
        {
            myKernelError[j] += myInputPadded[i + j] * outputError[i];
            myInputError[i] += outputErrorPadded[kernelSize() - 1 + i - j] * myKernel[j];
        }
    }
}

// -----------------------------------------------------------------------------
bool ConvLayer1D::optimize(const double learningRate)
{
    if (learningRate <= 0) { return false; }
    for (std::size_t i{}; i < kernelSize(); ++i)
    {
        myKernel[i] -= myKernelError[i] * learningRate;
    }
    return true;
}

// -----------------------------------------------------------------------------
void ConvLayer1D::initKernel(const std::size_t kernelSize)
{
    myKernel.resize(kernelSize);
    for (auto& i : myKernel)
    {
        i = utils::random<double>(0, 1);
    }
}

// -----------------------------------------------------------------------------
void ConvLayer1D::setInputPadded(const std::vector<double>& input)
{
    myInputPadded.resize(numPaddings() * 2 + input.size(), 0);
    for (std::size_t i{}; i < input.size(); ++i)
    {
        myInputPadded[i + numPaddings()] = input[i];
    }
}

// -----------------------------------------------------------------------------
std::size_t ConvLayer1D::numPaddings() const { return kernelSize() / 2; }

// -----------------------------------------------------------------------------
std::vector<double> ConvLayer1D::pad(const std::vector<double>& data)
{
    return utils::pad<double>(data, numPaddings());
}

} // namespace ml