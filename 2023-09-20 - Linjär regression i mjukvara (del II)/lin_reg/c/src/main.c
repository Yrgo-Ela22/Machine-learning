/********************************************************************************
 * @brief Implementation of linear regression model in C.
 ********************************************************************************/
#include <stdio.h>
#include <lin_reg.h>

/********************************************************************************
 * @brief Performs predictions with referenced regression model and prints the
 *        result in the terminal.
 * 
 * @param model
 *        Reference to regression model to perform prediction with.
 * @param inputs
 *        Reference to array storing the input values to predict with.
 * @param num_inputs
 *        The number of values in the input array.
 * @param ostream
 *        Reference to output stream for printing (default = terminal print).
 ********************************************************************************/
static void print_predictions(const struct lin_reg* model, 
                              const double* inputs, 
                              const size_t num_inputs,
                              FILE* ostream) {
    if (!ostream) ostream = stdout;
    fprintf(ostream, "--------------------------------------------------------------------------------\n");
    for (const double* i = inputs; i < inputs + num_inputs; ++i) {
        fprintf(ostream, "Input: %lg, Output: %lg\n", *i, lin_reg_predict(model, *i));
    }
    fprintf(ostream, "--------------------------------------------------------------------------------\n\n");
}

/********************************************************************************
 * @brief Creates linear regression model. The model is trained do detect the
 *        pattern y = 3x - 2 during 1000 epochs with a learning rate of 1 %. 
 *        The model is tested by predicting with the input values stored in 
 *        the input array. The result is printed in the terminal.
 ********************************************************************************/
int main(void) {
    struct lin_reg model;
    const double inputs[] = {0, 1, 2, 3, 4};
    const double outputs[] = {-2, 1, 4, 7, 10};
    lin_reg_init(&model, inputs, outputs, 5);
    lin_reg_train(&model, 1000, 0.01);
    print_predictions(&model, inputs, 5, 0);
    lin_reg_clear(&model);
    return 0;
}