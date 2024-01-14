/********************************************************************************
 * @brief Demonstrates function for printing arithmetic content of a vector
 *        on a single line, encircled in braces.
 ********************************************************************************/
#include <iostream>
#include <type_traits>
#include <vector>

namespace 
{

/********************************************************************************
 * @brief Prints numbers stored in vector on a single line, encircled in braces.
 * 
 * @tparam T      The vector type (must be arithmetic).
 * 
 * @param data    Reference to the vector whose content will be printed.
 * @param ostream Reference to output stream (default = terminal print).
 ********************************************************************************/
template <typename T = int>
void print(const std::vector<T>& data, std::ostream& ostream = std::cout) 
{
    static_assert(std::is_arithmetic<T>::value, "Non-arithmetic type selected for method ::Print!");
    const auto lastElement{&data[data.size() - 1]};
    ostream << "[";
    for (const auto& i : data) 
    {
        ostream << i;
        if (&i < lastElement) { ostream << ", "; }
    }
    ostream << "]\n";
}
} // namespace

/********************************************************************************
 * @brief Prints content of two vectors containing numbers of different types.
 * 
 * @return Success code 0 upon termination of the program.
 ********************************************************************************/
int main() 
{
    const std::vector<double> v1{1, 2, 3, 4, 5};
    const std::vector<int> v2{10, 20, 30};
    const std::vector<std::size_t> v3{125, 250, 500, 1000};
    
    print(v1);
    print(v2);
    print(v3);
    return 0;
}