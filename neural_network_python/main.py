from neural_network import NeuralNetwork
from utils import ActFunc

# ------------------------------------------------------------------------------
def main() -> None:
    trainInput = ((0, 0), (0, 1), (1, 0), (1, 1))
    trainOutput = ((0), (1), (1), (0))
    neuralNetwork = NeuralNetwork(2, 3, 1, ActFunc.Tanh)
    neuralNetwork.addTrainingData(trainInput, trainOutput)
    if neuralNetwork.train(200000, 0.1):
        neuralNetwork.printPredictions(trainInput)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()