/********************************************************************************
 * @brief Implementation of two-dimensional pooling layers to reduce the size
 *        of images while keeping the sharpness.
 ********************************************************************************/
#pragma once

#include <vector>

namespace ml
{

/********************************************************************************
 * @brief Enumeration class for selecting pooling type.
 * 
 * @param Max     Stores the most significant attributes.
 * @param Average Stores the average of the extracted attributes.
 ********************************************************************************/
enum class PoolType
{
    Max,
    Average
};

/********************************************************************************
 * @brief Class for implementation of two-dimensional pooling layers.
 *        The size of the images to pool is dynamic. Both max pooling and
 *        average pooling is supported.
 ********************************************************************************/
class PoolingLayer2D
{
public:

    /********************************************************************************
     * @brief Default constructor deleted.
     ********************************************************************************/
    PoolingLayer2D() = delete;

    /********************************************************************************
     * @brief Creates new pooling layer.
     * 
     * @param size The size of the pooling layer.
     * @param type The pooling type to use (default = max pooling).
     ********************************************************************************/
    PoolingLayer2D(const std::size_t size, const PoolType type = PoolType::Max);

    /********************************************************************************
     * @brief Provides the pooling layer output.
     * 
     * @return Reference to a vector holding the pooling layer output.
     ********************************************************************************/
    const std::vector<std::vector<double>>& output() const;

    /********************************************************************************
     * @brief Provides the pooling layer type.
     * 
     * @return The pooling layer type as an enumerator of enumeration class PoolType.
     ********************************************************************************/
    PoolType type() const;

     /********************************************************************************
     * @brief Provides the pooling layer size.
     * 
     * @return The size of the pooling layer as an unsigned integer.
     ********************************************************************************/
    std::size_t size() const;
    
     /********************************************************************************
     * @brief Performs pooling of referenced input image.
     * 
     * @param input Reference to image to pool.
     * 
     * @return True if pooling was performed.
     ********************************************************************************/
    bool feedforward(const std::vector<std::vector<double>>& input);

protected:
    std::size_t numPaddings() const;
    bool isInputInValid(const std::vector<std::vector<double>>& input);
    double pool(const std::vector<std::vector<double>>& input, 
                const std::size_t j, 
                const std::size_t i);
    double poolMax(const std::vector<std::vector<double>>& input, 
                   const std::size_t x, 
                   const std::size_t y);
    double poolAverage(const std::vector<std::vector<double>>& input, 
                       const std::size_t x, 
                       const std::size_t y);
    std::size_t iterationWidth(const std::vector<std::vector<double>>& input);
    std::size_t iterationHeight(const std::vector<std::vector<double>>& input);
    bool isWithinRange(const std::vector<std::vector<double>>& input, 
                       const std::size_t x, 
                       const std::size_t y);

    std::vector<std::vector<double>> myOutput{};
    const PoolType myType;
};

} // namespace ml