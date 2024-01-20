/********************************************************************************
 * @brief Implementation of one-dimensional convolutional layers to filter
 *        attributes from images.
 ********************************************************************************/
#pragma once 

#include <vector>

namespace ml
{

/********************************************************************************
 * @brief Class for implementation of one-dimensional convolutional layers.
 *        The size of the images to filter is dynamic. The number of strides
 *        is always set to one and padding is used, so the size of the filtered
 *        image is unchanged during feature extraction.
 ********************************************************************************/
class ConvLayer1D
{
public:

    /********************************************************************************
     * @brief Default constructor deleted.
     ********************************************************************************/
    ConvLayer1D() = delete; 

    /********************************************************************************
     * @brief Creates new convolutional layer.
     * 
     * @param kernelSize The size of the kernel used to filter the image.
     ********************************************************************************/
    ConvLayer1D(const std::size_t kernelSize);

    /********************************************************************************
     * @brief Provides the input image padded with zeros.
     * 
     * @return A reference to the padded input image.
     ********************************************************************************/
    const std::vector<double>& inputPadded() const;

    /********************************************************************************
     * @brief Provides the kernel used to filter the image.
     * 
     * @return A reference to the kernel.
     ********************************************************************************/
    const std::vector<double>& kernel() const;

    /********************************************************************************
     * @brief Provides the output of the convolutional layer, i.e. the attributes
     *        extracted from the input image.
     * 
     * @return A reference to the output of the convolutional layer.
     ********************************************************************************/
    const std::vector<double>& output() const;

    /********************************************************************************
     * @brief Provides the calculated kernel error used to optimize the layer.
     * 
     * @return A reference to the calculated kernel error.
     ********************************************************************************/
    const std::vector<double>& kernelError() const;

    /********************************************************************************
     * @brief Provides the calculated input error used to optimize the previous
     *        convolutional layer (if there is any).
     * 
     * @return A reference to the calculated input error.
     ********************************************************************************/
    const std::vector<double>& inputError() const;

    /********************************************************************************
     * @brief Provides the size of the last filtered image without padding.
     * 
     * @return The size of the last filtered image as an unsigned integer.
     ********************************************************************************/
    std::size_t imageSize() const;

    /********************************************************************************
     * @brief Provides the size of the kernel.
     * 
     * @return The kernel size an unsigned integer.
     ********************************************************************************/
    std::size_t kernelSize() const;

    /********************************************************************************
     * @brief Extracts features out of specified input image.
     * 
     * @param input Reference to the image to extract features from.
     ********************************************************************************/
    void feedforward(const std::vector<double>& input);

    /********************************************************************************
     * @brief Calculates kernel and input error for optimization.
     * 
     * @param outputError Calculated input error of the next convolutional layer.
     ********************************************************************************/
    void backpropagate(const std::vector<double>& outputError);

    /********************************************************************************
     * @brief Modifies the kernel parameters to increase the precision of the 
     *        feature extraction.
     * 
     * @param learningRate The adjustment rate of the kernel parameters.
     ********************************************************************************/
    bool optimize(const double learningRate = 0.01);

protected:

    void initKernel(const std::size_t kernelSize);
    void setInputPadded(const std::vector<double>& input);
    std::size_t numPaddings() const;

    void calculateKernelError();
    std::vector<double> pad(const std::vector<double>& data);

    std::vector<double> myInputPadded{};
    std::vector<double> myKernel{};
    std::vector<double> myOutput{};
    std::vector<double> myKernelError{};
    std::vector<double> myInputError{};
};

} // namespace ml