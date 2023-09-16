/********************************************************************************
 * @brief Implementation of linear regression model in C++.
 ********************************************************************************/
#include <iostream>
#include <lin_reg.hpp>

/********************************************************************************
 * @brief Creates linear regression model. The training data is passed when
 *        creating the model. Next lecture the model will be trained and will
 *        then be tested. 
 ********************************************************************************/
int main(void) {
    const std::vector<double> inputs{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<double> outputs{-2, 1, 4, 7, 10};
    yrgo::LinReg model{inputs, outputs};
    return 0;
}