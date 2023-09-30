/********************************************************************************
 * @brief Skeleton code for assignment in machine learning, where a neural
 *        node class is to be implemented.
 ********************************************************************************/
#include <vector>
#include <iostream>
#include <node.hpp>

namespace ml = yrgo::machine_learning;

namespace {

/********************************************************************************
 * @brief Provides the number of epochs to train the neural node with.
 *        If a number greater than 0 has been entered when running the program, 
 *        this value is used as the number of epochs to run, else the set
 *        default number of epochs is used.
 *        
 * @param argc 
 *        The number of arguments entered from the terminal.
 * @param argv 
 *        Array containing arguments entered from the terminal.
 * @param default_epochs
 *        The default number of epochs (default = 1000).
 * @return
 *        The number of epochs to train the neural node with.
 ********************************************************************************/
std::size_t GetNumEpochs(const int argc, 
                         const char** argv, 
                         const std::size_t default_epochs = 1000) {
    if (argc >= 2) {
        const auto num_epochs{std::atoi(argv[1])};
        return num_epochs > 0 ? num_epochs : default_epochs;
    } else {
        return default_epochs;
    }
}

} /* namespace */


/********************************************************************************
 * @brief Trains neural node to predict {1} when it's inputs are set to {1, 0}.
 *        The number of epochs to train is primarily entered from the terminal.
 *        If no value greater than 0 is entered, 1000 epochs used by default.
 *        The learning rate is set to 10 %. The node output is printed in the 
 *        terminal before terminating the program.
 *        
 * @param argc 
 *        The number of arguments entered from the terminal.
 * @param argv 
 *        Array containing arguments entered from the terminal.
 * @return
 *        Success code 0 upon termination of the program.
 ********************************************************************************/
int main(const int argc, const char** argv) {
    const std::vector<double> input{1, 0};
    const double reference{1};
    ml::Node n1{2};

    for (std::size_t i{}; i < GetNumEpochs(argc, argv); ++i) {
        n1.Feedforward(input);
        n1.Backpropagate(reference);
        n1.Optimize(input, 0.1);
    }
    std::cout << "Output: " << n1.Output() << "\n";
    return 0;
}