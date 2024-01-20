/********************************************************************************
 * @brief Implementation of two-dimensional convolutional layers to filter
 *        attributes from images.
 ********************************************************************************/
#pragma once

#include <vector>

namespace ml
{

/********************************************************************************
 * @brief Class for implementation of two-dimensional convolutional layers.
 *        The size of the images to filter is dynamic. The number of strides
 *        is always set to one and padding is used, so the size of the filtered
 *        image is unchanged during feature extraction.
 ********************************************************************************/
class ConvLayer2D
{
public:

    /********************************************************************************
     * @brief Default constructor deleted.
     ********************************************************************************/
    ConvLayer2D() = delete;

    /********************************************************************************
     * @brief Creates new convolutional layer.
     * 
     * @param kernelSize The size of the kernel used to filter the image.
     ********************************************************************************/
    ConvLayer2D(const std::size_t kernelSize);

    /********************************************************************************
     * @brief Provides the input image padded with zeros.
     * 
     * @return A reference to the padded input image.
     ********************************************************************************/
    const std::vector<std::vector<double>>& inputPadded() const;

    /********************************************************************************
     * @brief Provides the kernel used to filter the image.
     * 
     * @return A reference to the kernel.
     ********************************************************************************/
    const std::vector<std::vector<double>>& kernel() const;

    /********************************************************************************
     * @brief Provides the output of the convolutional layer, i.e. the attributes
     *        extracted from the input image.
     * 
     * @return A reference to the output of the convolutional layer.
     ********************************************************************************/
    const std::vector<std::vector<double>>& output() const;

    /********************************************************************************
     * @brief Provides the calculated kernel error used to optimize the layer.
     * 
     * @return A reference to the calculated kernel error.
     ********************************************************************************/
    const std::vector<std::vector<double>>& inputError() const;

    /********************************************************************************
     * @brief Provides the calculated input error used to optimize the previous
     *        convolutional layer (if there is any).
     * 
     * @return A reference to the calculated input error.
     ********************************************************************************/
    const std::vector<std::vector<double>>& kernelError() const;

    /********************************************************************************
     * @brief Provides the image width.
     * 
     * @return The image width an unsigned integer.
     ********************************************************************************/
    std::size_t imageWidth() const;

     /********************************************************************************
     * @brief Provides the image height.
     * 
     * @return The image height an unsigned integer.
     ********************************************************************************/
    std::size_t imageHeight() const;

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
    void feedforward(const std::vector<std::vector<double>>& input);

    /********************************************************************************
     * @brief Calculates kernel and input error for optimization.
     * 
     * @param outputError Calculated input error of the next convolutional layer.
     ********************************************************************************/
    void backpropagate(const std::vector<std::vector<double>>& outputError);

    /********************************************************************************
     * @brief Modifies the kernel parameters to increase the precision of the 
     *        feature extraction.
     * 
     * @param learningRate The adjustment rate of the kernel parameters.
     ********************************************************************************/
    void optimize(const double learningRate = 0.01);

protected:
    void initKernel(const std::size_t kernelSize);
    std::vector<std::vector<double>> pad(const std::vector<std::vector<double>>& input);
    std::size_t numPaddings() const;

    static std::size_t width(const std::vector<std::vector<double>>& input);
    static std::size_t height(const std::vector<std::vector<double>>& input);

    std::vector<std::vector<double>> myInputPadded{};
    std::vector<std::vector<double>> myKernel{};
    std::vector<std::vector<double>> myOutput{};
    std::vector<std::vector<double>> myInputError{};
    std::vector<std::vector<double>> myKernelError{};
};

} // namespace ml