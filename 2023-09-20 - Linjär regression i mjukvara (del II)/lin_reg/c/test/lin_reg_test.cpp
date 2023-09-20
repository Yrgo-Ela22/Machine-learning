/********************************************************************************
 * @brief Testing linear regression model implementation with Google Test.
 ********************************************************************************/
#include <gtest/gtest.h>
#include <lin_reg.h>

/********************************************************************************
 * @brief Tests model trained to predict y = 2x + 2 during 1000 epochs with
 *        a learning rate of 1 %.
 ********************************************************************************/
TEST(LinRegTest, Test1) { 
    const std::vector<double> inputs{0, 1, 2, 3, 4};
    const std::vector<double> outputs{2, 4, 6, 8, 10};
    lin_reg model{};
    lin_reg_init(&model, inputs.data(), outputs.data(); inputs.size());
    lin_reg_train(&model, 1000, 0.01);
    for (std::size_t i{}; i < inputs.size(); ++i) {
        EXPECT_NEAR(outputs[i], lin_reg_predict(&model, inputs.data()[i]), 0.001); 
    }
    lin_reg_clear(&model);
}

/********************************************************************************
 * @brief Tests model trained to predict y = 3x - 5 during 1000 epochs with
 *        a learning rate of 1 %.
 ********************************************************************************/
TEST(LinRegTest, Test2) { 
    const std::vector<double> inputs{0, 1, 2, 3, 4};
    const std::vector<double> outputs{-5, -2, 1, 4, 7};
    lin_reg model{};
    lin_reg_init(&model, inputs.data(), outputs.data(); inputs.size());
    lin_reg_train(&model, 1000, 0.01);
    for (std::size_t i{}; i < inputs.size(); ++i) {
        EXPECT_NEAR(outputs[i], lin_reg_predict(&model, inputs.data()[i]), 0.001); 
    }
    lin_reg_clear(&model);
}

/********************************************************************************
 * @brief Tests model trained to predict y = 2.5x - 10 during 1000 epochs with
 *        a learning rate of 1 %.
 ********************************************************************************/
TEST(LinRegTest, Test3) { 
    const std::vector<double> inputs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<double> outputs{-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5};
    lin_reg model{};
    lin_reg_init(&model, inputs.data(), outputs.data(); inputs.size());
    lin_reg_train(&model, 1000, 0.01);
    for (std::size_t i{}; i < inputs.size(); ++i) {
        EXPECT_NEAR(outputs[i], lin_reg_predict(&model, inputs.data()[i]), 0.001); 
    }
    lin_reg_clear(&model);
}

/********************************************************************************
 * @brief Initializes Google Test framework and runs all tests.
 * 
 * @param argc
 *        The number of input arguments entered rom the terminal when running 
 *        the program.
 * @param argv
 *        Reference to vector storing arguments input arguments as strings. 
 * @return
 *        Success code 0 if all tests were successful, else error code 1.
 ********************************************************************************/
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
}