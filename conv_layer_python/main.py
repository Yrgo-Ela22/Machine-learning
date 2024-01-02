from conv_layer import ConvLayer2D, printMatrix
from pooling_layer import PoolingLayer2D
from flatten_layer import FlattenLayer

# ------------------------------------------------------------------------------
def main() -> None:

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    outputError = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    convLayer = ConvLayer2D(2)
    poolingLayer = PoolingLayer2D(2)
    flattenLayer = FlattenLayer()

    convLayer.feedforward(matrix)
    convLayer.backpropagate(outputError)
    convLayer.optimize(0.001)
    convLayer.feedforward(matrix)

    poolingLayer.feedforward(convLayer.output())
    flattenLayer.feedforward(poolingLayer.output())

    printMatrix(convLayer.output())
    printMatrix(poolingLayer.output())
    flattenLayer.print()

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()