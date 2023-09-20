/********************************************************************************
 * @brief Implementation of linear regression model in C++.
 ********************************************************************************/
#include <iostream>
#include <iomanip>
#include <lin_reg.hpp>

namespace {

/********************************************************************************
 * @brief Performs predictions with referenced regression model and prints the
 *        result in the terminal.
 * 
 * @param model
 *        Reference to regression model to perform prediction with.
 * @param inputs
 *        Reference to vector storing the input values to predict with.
 * @param ostream
 *        Reference to output stream for printing (default = terminal print).
 * @param num_decimals
 *        The number of decimals to print with (default = 1).
 ********************************************************************************/
void PrintPredictions(const yrgo::LinReg& model, 
                      const std::vector<double>& inputs, 
                      std::ostream& ostream = std::cout,
                      const std::size_t num_decimals = 1) {
    ostream << std::fixed << std::setprecision(num_decimals);
    ostream << "--------------------------------------------------------------------------------\n";
    for (auto& input : inputs) {
        ostream << "Input: " << input << ", Output: " << model.Predict(input) << "\n";
    }
    ostream << "--------------------------------------------------------------------------------\n\n";
}
} /* namespace */

/********************************************************************************
 * @brief Creates linear regression model. The model is trained do detect the
 *        pattern y = 3x - 2 during 1000 epochs with a learning rate of 1 %. 
 *        The model is tested by predicting with the input values stored in 
 *        the input vector. The result is printed in the terminal.
 ********************************************************************************/
int main(void) {
    const std::vector<double> inputs{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<double> outputs{-2, 1, 4, 7, 10};
    yrgo::LinReg model{inputs, outputs};
    model.Train(1000);
    PrintPredictions(model, inputs);
    return 0;
}