/********************************************************************************
 * @brief Contains class for implementation of dense layers.
 ********************************************************************************/
#pragma once

#include <vector>
#include <utils.hpp>

namespace yrgo {
namespace machine_learning {

/********************************************************************************
 * @brief Enumeration for selecting activation function between ReLU and tanH.
 *
 * @param kRelu Enumerator for selecting ReLU (Rectified Linear Unit)-
 * @param ktanh Enumerator for selecting Tanh (the hyperbolic tangent function).
 ********************************************************************************/
enum class ActFunc { kRelu, kTanh };

class DenseLayer {
  public:
    /********************************************************************************
     * @brief Creates empty dense layer.
     ********************************************************************************/
    DenseLayer(void) = default;

    /********************************************************************************
    * @brief Creates new dense layer with specified number of nodes and weights.
    * 
    * @param num_nodes            The number of nodes in the layer.
    * @param num_weights_per_node The number of weights per node in the layer.
    * @param act_func             Activation function (default = ReLU).
    ********************************************************************************/
   DenseLayer(const std::size_t num_nodes, 
              const std::size_t num_weights_per_node,
              const enum ActFunc act_func = ActFunc::kRelu);

  private:
    std::vector<double> output_{};               /* Holds output values. */
    std::vector<double> bias_{};                 /* Holds bias values. */
    std::vector<double> error_{};                /* Holds calculated errors. */
    std::vector<std::vector<double>> weights_{}; /* Holds weights for all nodes. */
    enum ActFunc act_func_{ActFunc::kRelu};      /* Selected activation function. */
};

} /* namespace machine_learning */
} /* namespace yrgo */