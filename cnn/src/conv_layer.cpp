#include <conv_layer.hpp>

namespace yrgo {
namespace machine_learning {

ConvLayer::ConvLayer(const size_t image_size, const size_t kernel_size) { 

}

bool ConvLayer::Feedforward(const std::vector<std::vector<double>>& input) {
    return false;
}

void ConvLayer::PrintMatrix(const std::vector<std::vector<double>>& data,
                            std::ostream& ostream,
                            const int num_decimals,
                            const size_t offset) {
    for (size_t i{offset}; i < data.size() - offset; ++i) {
        for (size_t j{offset}; j < data.size() - offset; ++j) {
            ostream << std::setprecision(num_decimals) << data[j][i] << " ";
        }
        ostream << "\n";
    }
}

void ConvLayer::Print(std::ostream& ostream, const int num_decimals) {
    if (image_.size() == 0) return;
    ostream << std::fixed;
    ostream << "------------------------------------------------------------------------------\n";
    ostream << "Image size: " << image_.size() - 2 << " x " << image_.size() - 2 << "\n";
    ostream << "Kernel size: " << kernel_.size() << " x " << kernel_.size() << "\n\n";

    ostream << "Image:\n";
    PrintMatrix(image_, ostream, num_decimals, 1);
    ostream << "\nKernel:\n";
    PrintMatrix(kernel_, ostream, num_decimals);

    ostream << "\nFeature map:\n";
    PrintMatrix(output_, ostream, num_decimals);
    ostream << "------------------------------------------------------------------------------\n\n";
}

} /* namespace machine_learning */
} /* namespace yrgo */