/********************************************************************************
 * @brief Implementation details for the LinReg class.
 ********************************************************************************/
#include <lin_reg.hpp>

namespace yrgo {

/********************************************************************************
 * @note  Implementation details:
 *        1. The specified training data is copied and stored in vectors.
 *        2. If the number of input and reference values don't match, the
 *           superfluous values are deleted by resizing the corresponding vector.
 *        3. The index of each training set is stored in the train order vector.
 *        4. The random generator is initializes to ensure that the randomized
 *           sequence is unique each time the program is run.
 ********************************************************************************/
void LinReg::LoadTrainingData(const std::vector<double>& train_in, 
                              const std::vector<double>& train_out) {
    train_in_ = train_in; 
    train_out_ = train_out;
    MatchTrainingSets();
    InitTrainOrderVector();
    InitRandomGenerator();
}

/********************************************************************************
 * @note Implementation details:
 *        1. A loop is generated to run num_epochs number of times.
 *        2. The training order is randomized before the training begins.
 *        3. We train the model with all the training sets one by one.
 *        4. We fetch the index of the training set and optimize out model.
 ********************************************************************************/
void LinReg::Train(const std::size_t num_epochs, const double learning_rate) {
    for (std::size_t i{}; i < num_epochs; ++i) {
        RandomizeTrainingOrder();
        for (auto& j : train_order_) { 
            Optimize(train_in_[j], train_out_[j], learning_rate);
        }
    }
}

/********************************************************************************
 * @note  Implementation details:
 *        1. We iterate through the training order vector.
 *        2. We generate a random index between 0 - train_order.size() - 1.
 *           For instance, if the training order vector has five elements, we
 *           generate a random number between 0 - 4.
 *        3. We swap the values of index i and r in the vector.
 ********************************************************************************/
void LinReg::RandomizeTrainingOrder(void) {
    for (std::size_t i{}; i < train_order_.size(); ++i) {
        const auto r{std::rand() % train_order_.size()}; 
        const auto temp{train_order_[i]};
        train_order_[i] = train_order_[r];
        train_order_[r] = temp;
    }
}

/********************************************************************************
 * @note  Implementation details:
 *        1. If input != 0, we predict with the input and optimize according
 *           to the error.
 *        2. Else, we set the bias to the y_ref value, since y = m if x = 0.
 *           (y = kx + m = k * 0 + 0 => y = m when k = 0).
 ********************************************************************************/
void LinReg::Optimize(const double input, const double reference, const double learning_rate) {
    if (input != 0) {
        const auto error{reference - Predict(input)}; /* error = y_ref - y_pred */
        bias_ += error * learning_rate;               /* m = m + error * LR */
        weight_ += error * learning_rate * input;     /* k = k + error * LR * x */
    } else {
        bias_ = reference;                            /* m = y_ref when x = 0 */
    }
}

/********************************************************************************
 * @note  Implementation details:
 *        1. If the number of input and reference values don't match, i.e. the
 *           size of the train_in_ and train_out_ vectors don't match, the
 *           superfluous values are removed by resizing the larger vector to the
 *           size of the smaller one.
 ********************************************************************************/
void LinReg::MatchTrainingSets(void) {
    if (train_in_.size() != train_out_.size()) {
        const auto num_sets{train_in_.size() < train_out_.size() ? 
                            train_in_.size() : train_out_.size()};
        train_in_.resize(num_sets);
        train_out_.resize(num_sets);
    }
}

/********************************************************************************
 * @note  Implementation details:
 *        1. The size of the train order vector is set to the number of stored
 *           training sets.
 *        2. The vector is assigned the index of each stored training set, e.g.
 *           0 - 9 if ten training sets are stored.
 ********************************************************************************/
void LinReg::InitTrainOrderVector(void) {
    train_order_.resize(train_in_.size());
    for (std::size_t i{}; i < train_order_.size(); ++i) {
        train_order_[i] = i;
    }
}

/********************************************************************************
 * @note  Implementation details:
 *        1. The size of the train order vector is set to the number of stored
 *           training sets.
 *        2. The vector is assigned the index of each stored training set, e.g.
 *           0 - 9 if ten training sets are stored.
 ********************************************************************************/
void LinReg::InitRandomGenerator(void) {
    static bool random_generator_initialized{false};
    if (!random_generator_initialized) {
        std::srand(std::time(nullptr));
        random_generator_initialized = true;
    }
}

} /* namespace yrgo */