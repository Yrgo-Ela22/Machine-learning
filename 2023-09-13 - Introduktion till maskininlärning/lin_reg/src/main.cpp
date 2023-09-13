/********************************************************************************
 * @brief Implementation of linear regression model in C++.
 ********************************************************************************/
#include <iostream>
#include <lin_reg.hpp>

/********************************************************************************
 * @brief Creates empty regression model and stores training data in vectors. 
 *        Next lecture the model will be initialized with the training data 
 *        and training will be performed.
 ********************************************************************************/
int main(void) {
    const std::vector<double> inputs{0, 1, 2, 3, 4};
    const std::vector<double> outputs{-2, 1, 4, 7, 10};
    yrgo::LinReg model{};
    return 0;
}