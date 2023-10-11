/********************************************************************************
 * @brief Implementation details for the DenseLayer class.
 ********************************************************************************/
#include <dense_layer.hpp>

namespace yrgo {
namespace machine_learning {

/********************************************************************************
 * @note The function is implemented as follows:
 * 
 *       1. Initializes the random generator.
 *       3. Initializes the output vector, starting values = 0.
 *       3. Initializes the bias vector, starting values = random[0,1].
 *       4. Initializes the error vector, starting values = 0.
 *       5. Initializes the weight vector, starting values = random[0,1].
 *       6. Sets the activation function.
 ********************************************************************************/
DenseLayer::DenseLayer(const std::size_t num_nodes, 
                       const std::size_t num_weights_per_node,
                       const enum ActFunc act_func) {
    utils::random::Init();
    output_.resize(num_nodes, 0); 
    utils::random::InitVector<double>(bias_, num_nodes, 0, 1);
    error_.resize(num_nodes, 0);
    utils::random::InitVector<double>(weights_, num_nodes, num_weights_per_node, 0, 1);
    act_func_ = act_func;
} 

/********************************************************************************
 * @note The function is implemented as follows:
 *
 *       1. We iterate through the nodes (from first to last).
 *       2. We add bias + the sum of inputs * weights for the node.
 *       3. We pass the sum through the activation function and stores the
 *          resulting output.
 ********************************************************************************/
void DenseLayer::Feedforward(const std::vector<double>& inputs) {
    for (std::size_t i{}; i < NumNodes(); ++i) {
        double sum{bias_[i]};
        for (std::size_t j{}; j < NumWeightsPerNode() && j < inputs.size(); ++j) {
            sum += inputs[j] * weights_[i][j];
        }
        output_[i] = act_func_ == ActFunc::kRelu ? 
            utils::math::Relu(sum) : utils::math::Tanh(sum);
    }
}

/********************************************************************************
 * @note The function is implemented as follows:
 * 
 *       1. We iterate through the nodes (from first to last).
 *       2. We calculate the error for each node as the difference between the
 *          reference value and the output value.
 *       3. We save the error as the calculated deviation multiplied with the
 *          derivate of the output. Depending on selected activation function
 *          we either multiply with the derivate of ReLU or tanh.
 ********************************************************************************/
void DenseLayer::Backpropagate(const std::vector<double>& reference) {
    for (std::size_t i{}; i < NumNodes(); ++i) {
        const double error = reference[i] - output_[i];
        error_[i] = act_func_ == ActFunc::kRelu ? error * utils::math::ReluDelta(output_[i]) 
                                                : error * utils::math::TanhDelta(output_[i]);
    }
}

/********************************************************************************
 * @note The function is implemented as follows:
 * 
 *       1. We iterate through the nodes (from first to last).
 *       2. We adjust bias by adding the error multiplied with the learning rate.
 ********************************************************************************/
void DenseLayer::Optimize(const std::vector<double>& inputs, const double learning_rate) {
    for (std::size_t i{}; i < NumNodes(); ++i) {
        bias_[i] += error_[i] * learning_rate;
        for (std::size_t j{}; j < NumWeightsPerNode() && j < inputs.size(); ++j) {
            weights_[i][j] += error_[i] * learning_rate * inputs[j];
        }
    }
}


} /* namespace machine_learning */
} /* namespace yrgo */
