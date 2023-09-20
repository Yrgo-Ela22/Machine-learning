/********************************************************************************
 * @brief Library for implementing linear regression models in C.
 ********************************************************************************/
#pragma once

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

/********************************************************************************
 * @brief Struct for implementing linear regression models.
 ********************************************************************************/
struct lin_reg {
    const double* train_in;  /* Pointer to array storing input values (x). */
    const double* train_out; /* Pointer to array storing reference values (y_ref). */
    size_t* train_order;     /* Pointer to array storing indexes for training sets. */
    size_t num_sets;         /* The number of stored training sets. */
    double weight;           /* k-value. */
    double bias;             /* m-value. */
};

/********************************************************************************
 * @brief Initializes regression model and stores referenced training data.
 * 
 * @param self
 *        Reference to the regression model to initialize.
 * @param train_in
 *        Reference to array containing input data (x).
 * @param train_out
 *        Reference to array containing reference data (y_ref).
 * @param num_sets
 *        The number of training sets in the arrays.
 * @return
 *        True if the initialization succeeded, else false.
 ********************************************************************************/
bool lin_reg_init(struct lin_reg* self, 
                  const double* train_in, 
                  const double* train_out, 
                  const size_t num_sets);

/********************************************************************************
 * @brief Clears parameters for linear regression model and frees dynamically 
 *        memory for the training order array.
 * 
 * @param self
 *        Reference to the regression model to initialize.
 ********************************************************************************/
void lin_reg_clear(struct lin_reg* self);

/********************************************************************************
 * @brief Makes a prediction with the specified input value. 
 *        The prediction is calculated as 
 * 
 *                                y_pred = kx + m,
 *  
 * where x is the specified input value, k is the weight and m is the bias.
 *        
 * @param self
 *        Reference to the regression model to predict with.
 * @param input
 *        The input value (x) to predict with.
 * @return
 *        The predicted value (y_pred).
 ********************************************************************************/
static double lin_reg_predict(const struct lin_reg* self, const double input) { 
    return self->weight * input + self->bias; 
}


/********************************************************************************
 * @brief Trains regression model with specified parameters.
 * 
 * @param self
 *        Reference to the regression model to predict with.
 * @param num_epochs
 *        The number of epochs (turns) to train.
 * @param learning_rate
 *        The learning rate, sets the change rate during errors.
 ********************************************************************************/
void lin_reg_train(struct lin_reg* self, 
                   const size_t num_epochs, 
                   const double learning_rate);

#ifdef __cplusplus
}
#endif
