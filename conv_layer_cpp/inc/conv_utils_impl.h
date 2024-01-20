/********************************************************************************
 * @brief Implementation details function templates in ml::utils.
 ********************************************************************************/
#pragma once

namespace ml
{
namespace utils
{
namespace 
{

// -----------------------------------------------------------------------------
constexpr std::size_t numPaddings(const std::size_t kernelSize)
{
    return static_cast<std::size_t>(kernelSize / 2);    
}

// -----------------------------------------------------------------------------
template <typename T>
std::size_t numPaddings(const std::vector<T>& kernel)
{
    static_assert(std::is_arithmetic<T>::value);
    return numPaddings(kernel.size());
}

// -----------------------------------------------------------------------------
template <typename T = double>
std::size_t numPaddings(const std::vector<std::vector<T>>& kernel)
{
    static_assert(std::is_arithmetic<T>::value);
    return numPaddings(kernel.size());
}

// -----------------------------------------------------------------------------
template <typename T>
std::vector<T> pad(const std::vector<T>& data, 
                   const std::size_t numPaddings, 
                   const T padValue)
{
    static_assert(std::is_arithmetic<T>::value, 
        "Function ml::pad does not support non-arithmetic types!");
    std::vector<T> padded(numPaddings * 2 + data.size(), padValue);
    for (std::size_t i{}; i < data.size(); ++i)
    {
        padded[numPaddings + i] = data[i];
    }
    return padded;
}

// -----------------------------------------------------------------------------
template <typename T>
std::vector<T> pad(const std::vector<T>& data, 
                   const std::vector<T>& kernel,
                   const T padValue)
{
    static_assert(std::is_arithmetic<T>::value, 
        "Function ml::pad does not support non-arithmetic types!");
    return pad<T>(data, numPaddings(kernel), padValue);
}

// -----------------------------------------------------------------------------
template <typename T>
std::vector<std::vector<T>> pad(const std::vector<std::vector<T>>& data, 
                                const std::size_t numPaddings,
                                const T padValue)
{
    static_assert(std::is_arithmetic<T>::value, 
        "Function ml::pad does not support non-arithmetic types!");
    if (data.empty()) { return {}; }
    const auto numValuesPerRow(numPaddings * 2 + data[0].size());
    std::vector<std::vector<double>> padded(numPaddings * 2 + data.size(), 
        std::vector<double>(numValuesPerRow, padValue));
    for (std::size_t i{}; i < data.size(); ++i)
    {
        for (std::size_t j{}; j < data[0].size(); ++j)
        {
            padded[numPaddings + i][numPaddings + j] = data[i][j];
        }
    }
    return padded;
}

// -----------------------------------------------------------------------------
template <typename T>
std::vector<std::vector<T>> pad(const std::vector<std::vector<T>>& data, 
                                const std::vector<std::vector<double>>& kernel,
                                const T padValue)
{
    static_assert(std::is_arithmetic<T>::value, 
        "Function ml::pad does not support non-arithmetic types!");
    return pad(data, numPaddings(kernel), padValue);
}

// -----------------------------------------------------------------------------
template <typename T>
void print(const std::vector<T>& data, 
           const std::size_t numDecimals, 
           std::ostream& ostream,
           const char* const end)
{
    static_assert(std::is_arithmetic<T>::value, 
        "Function ml::print does not support non-arithmetic types!");
    ostream << std::fixed << std::setprecision(numDecimals);
    ostream << "[";
    for (const auto& i : data)
    {
        ostream << i;
        if (&i != &data[data.size() - 1]) { ostream << ", "; }
    }
    ostream << "]";
    if (end) { ostream << end; }
}

// -----------------------------------------------------------------------------
template <typename T>
void print(const std::vector<std::vector<T>>& data, 
           const std::size_t numDecimals, 
           std::ostream& ostream)
{
    static_assert(std::is_arithmetic<T>::value, 
        "Function ml::print does not support non-arithmetic types!");
    ostream << std::fixed << std::setprecision(numDecimals);
    ostream << "--------------------------------------------------------------------------------\n";
    for (const auto& i : data) { print(i, numDecimals, ostream); }
    ostream << "--------------------------------------------------------------------------------\n\n";
}

// -----------------------------------------------------------------------------
inline void initRandomGenerator() 
{ 
    static bool randomGeneratorInitialized{false};
    if (!randomGeneratorInitialized)
    {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        randomGeneratorInitialized = true;
    }
}

// -----------------------------------------------------------------------------
template <typename T = int>
enable_if_integral<T, T> random(const T min, const T max)
{
    static_assert(std::is_integral<T>::value);
    return std::rand() % (max + 1 - min) - min; 
}

// -----------------------------------------------------------------------------
template <typename T = double>
enable_if_float<T, T> random(const T min, const T max)
{
    static_assert(std::is_floating_point<T>::value);
    return (static_cast<T>(std::rand()) / RAND_MAX) * (max - min) - min;
}

} // namespace
} // namespace utils
} // namespace ml