from __future__ import annotations
from utils import ActFunc
import utils

class DenseLayer:
    
    # ------------------------------------------------------------------------------
    def __init__(self, numNodes: int, numWeightsPerNode: int, actFunc: ActFunc = ActFunc.Relu) -> None:
        self._output = []
        self._error = []
        self._bias = []
        self._weights = []
        self._actFunc = actFunc
        self._initNodes(numNodes, numWeightsPerNode)

    # ------------------------------------------------------------------------------
    def output(self) -> tuple[float]:
        return tuple(self._output)
    
    # ------------------------------------------------------------------------------
    def error(self) -> tuple[float]:
        return tuple(self._error)
    
    # ------------------------------------------------------------------------------
    def bias(self) -> tuple[float]:
        return tuple(self._bias)
    
    # ------------------------------------------------------------------------------
    def weights(self) -> tuple[list[float]]:
        return tuple(self._weights)
    
    # ------------------------------------------------------------------------------
    def actFunc(self) -> ActFunc:
        return self._actFunc
    
    # ------------------------------------------------------------------------------
    def numNodes(self) -> int: 
        return len(self._output)
    
    # ------------------------------------------------------------------------------
    def numWeightsPerNode(self) -> int:
        return len(self._weights[0]) if len(self._weights) > 0 else 0
    
    # ------------------------------------------------------------------------------
    def feedforward(self, input: tuple[float]) -> None:
        for i in range(self.numNodes()):
            value = self._bias[i]
            for j in range(self.numWeightsPerNode()):
                if j < len(input):
                    value += input[j] * self._weights[i][j]
            self._output[i] = utils.actFuncOutput(value, self._actFunc)

    # ------------------------------------------------------------------------------
    def backpropagateOutput(self, output: tuple[float]) -> None:
        for i in range(self.numNodes()):
            if i < len(output):
                error = output[i] - self._output[i]
                self._error[i] = error * utils.actFuncDelta(self._output[i], self._actFunc)

    # ------------------------------------------------------------------------------
    def backpropagateHidden(self, nextLayer: DenseLayer) -> None:
        for i in range(self.numNodes()):
            error = 0.0
            for j in range(nextLayer.numNodes()):
                error += nextLayer.error()[j] * nextLayer.weights()[j][i]
            self._error[i] = error * utils.actFuncDelta(self._output[i], self._actFunc)

    # ------------------------------------------------------------------------------
    def optimize(self, input: tuple[float], learningRate: float) -> None:
        for i in range(self.numNodes()):
            self._bias[i] += self._error[i] * learningRate
            for j in range(self.numWeightsPerNode()):
                if j < len(input):
                    self._weights[i][j] += self._error[i] * learningRate * input[j]

    # ------------------------------------------------------------------------------
    def _initNodes(self, numNodes: int, numWeightsPerNode: int) -> None:
        utils.initList(self._output, numNodes, 0)
        utils.initList(self._error, numNodes, 0)
        utils.initRandomList(self._bias, numNodes)
        utils.initRandom2DList(self._weights, numNodes, numWeightsPerNode)