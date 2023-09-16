/********************************************************************************
 * @note Implementation details for the LinReg class.
 ********************************************************************************/
#include <lin_reg.hpp>

namespace yrgo {

/********************************************************************************
 * @brief The function is implemented as follows: 
 *        1. The specified training data is copied and stored in vectors.
 *        2. If the number of input and reference values don't match, the
 *           superfluous values are deleted by resizing the corresponding vector.
 *        3. The index of each training set is stored in the train order vector.
 ********************************************************************************/
void LinReg::LoadTrainingData(const std::vector<double>& train_in, 
                              const std::vector<double>& train_out) {
    train_in_ = train_in; 
    train_out_ = train_out;
    if (train_in_.size() != train_out_.size()) {
        const auto num_sets{train_in_.size() < train_out_.size() ? 
                   train_in_.size() : train_out_.size()};
        train_in_.resize(num_sets);
        train_out_.resize(num_sets);
    }
    train_order_.resize(train_in_.size());
    for (std::size_t i{}; i < train_order_.size(); ++i) {
        train_order_[i] = i;
    }
}

void LinReg::Train(const std::size_t num_epochs, const double learning_rate) {

}


} /* namespace yrgo */