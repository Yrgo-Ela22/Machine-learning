/********************************************************************************
 * @brief Library for implementing linear regression models.
 ********************************************************************************/
#pragma once

#include <vector>
#include <cstdlib>
#include <iostream>

namespace yrgo {

/********************************************************************************
 * @brief Class for implementing linear regression models.
 ********************************************************************************/
class LinReg {
  public:

    /********************************************************************************
     * @brief Default constructor, creates empty regression model.
     ********************************************************************************/
    LinReg(void) = default;

    /********************************************************************************
     * @brief Creates new regression model and stores referenced training data.
     * 
     * @param train_in
     *        Reference to vector containing input data (x).
     * @param train_out
     *        Reference to vector containing reference data (y_ref).
     ********************************************************************************/
    LinReg(const std::vector<double>& train_in, const std::vector<double>& train_out) {
        LoadTrainingData(train_in, train_out);
    }

    /********************************************************************************
     * @brief Loads training data from referenced vectors.
     * 
     * @param train_in
     *        Reference to vector containing input data (x).
     * @param train_out
     *        Reference to vector containing reference data (y_ref).
     ********************************************************************************/
    void LoadTrainingData(const std::vector<double>& train_in, 
                          const std::vector<double>& train_out);

    /********************************************************************************
     * @brief Trains regression model with specified parameters.
     * 
     * @param num_epochs
     *        The number of epochs (turns) to train.
     * @param learning_rate
     *        The learning rate, sets the change rate during errors (default = 0.01).
     ********************************************************************************/
    void Train(const std::size_t num_epochs, const double learning_rate = 0.01);

  /********************************************************************************
   * @note The private segment is only visible internally (i.e. in this class).
   ********************************************************************************/
  private:
    std::vector<double> train_in_{};         /* Input values (x). */
    std::vector<double> train_out_{};        /* Reference values (y_ref). */
    std::vector<std::size_t> train_order_{}; /* Stores indexes for training sets. */
    double weight_{};                        /* k-value. */
    double bias_{};                          /* m-value. */
};

} /* namespace yrgo */