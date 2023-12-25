from dense_layer import DenseLayer
from utils import ActFunc
import utils

class NeuralNetwork:

    # ------------------------------------------------------------------------------
    def __init__(self, 
                 numInputs: int, 
                 numHiddenNodes: int, 
                 numOutputs: int,
                 actFuncHidden: ActFunc = ActFunc.Relu,
                 actFuncOutput: ActFunc = ActFunc.Relu) -> None:
        self._hiddenLayer = DenseLayer(numHiddenNodes, numInputs, actFuncHidden)
        self._outputLayer = DenseLayer(numOutputs, numHiddenNodes, actFuncOutput)
        self._trainInput = []
        self._trainOutput = []
        self._trainOrder = []

    # ------------------------------------------------------------------------------ 
    def numTrainingSets(self) -> int:
        return len(self._trainOrder)

    # ------------------------------------------------------------------------------ 
    def output(self) -> tuple[float]:
        return self._outputLayer.output()

    # ------------------------------------------------------------------------------ 
    def addTrainingData(self, trainInput: tuple[tuple[float]], trainOutput: tuple[tuple[float]]) -> bool:  
        self._trainInput = utils.toContainer(trainInput)
        self._trainOutput = utils.toContainer(trainOutput)
        self._checkTrainingData()
        self._initTrainOrderList()
        return self.numTrainingSets() > 0

    # ------------------------------------------------------------------------------ 
    def train(self, numEpochs: int, learningRate: float = 0.01) -> bool:
        if self.numTrainingSets() == 0 or numEpochs <= 0 or learningRate <= 0: 
            return False
        for i in range(numEpochs):
            self._randomizeTrainingOrder()
            for j in self._trainOrder:
                input = utils.toContainer(self._trainInput[j])
                output = utils.toContainer(self._trainOutput[j])
                self._feedforward(input)
                self._backpropagate(output)
                self._optimize(input, learningRate)
        return True
    
    # ------------------------------------------------------------------------------ 
    def predict(self, input: tuple[float]) -> tuple[float]:
        self._feedforward(input)
        return self.output()
    
    # ------------------------------------------------------------------------------ 
    def printPredictions(self, inputSets: tuple[tuple[float]], numDecimals: int = 1) -> None:
        if len(inputSets) == 0 or numDecimals < 0: return
        print("--------------------------------------------------------------------------------")
        for input in inputSets:
            print("Input:", end="\t")
            utils.printFloats(input, numDecimals)
            print("Output:", end="\t")
            utils.printFloats(self.predict(input), numDecimals)  
            if (input != inputSets[len(inputSets) - 1]): print()
        print("--------------------------------------------------------------------------------\n")

    # ------------------------------------------------------------------------------ 
    def _checkTrainingData(self) -> None:
        if len(self._trainInput) != len(self._trainOutput):
            numSets = utils.min(len(self._trainInput), len(self._trainOutput))
            self._trainInput = self._trainInput[:numSets]
            self._trainOutput = self._trainOutput[:numSets]

    # ------------------------------------------------------------------------------ 
    def _initTrainOrderList(self) -> None:
        for i in range(len(self._trainInput)):
            self._trainOrder.append(i)

    # ------------------------------------------------------------------------------ 
    def _randomizeTrainingOrder(self) -> None:
        import random
        for i in range(self.numTrainingSets()):
            r = random.randint(0, len(self._trainOrder) - 1)
            temp = self._trainOrder[i]
            self._trainOrder[i] = self._trainOrder[r]
            self._trainOrder[r] = temp

    # ------------------------------------------------------------------------------ 
    def _feedforward(self, input: tuple[float]) -> None:
        self._hiddenLayer.feedforward(input)
        self._outputLayer.feedforward(self._hiddenLayer.output())

    # ------------------------------------------------------------------------------ 
    def _backpropagate(self, output: tuple[float]) -> None:
        self._outputLayer.backpropagateOutput(output)
        self._hiddenLayer.backpropagateHidden(self._outputLayer)

    # ------------------------------------------------------------------------------ 
    def _optimize(self, input: tuple[float], learningRate: float) -> None:
        self._hiddenLayer.optimize(input, learningRate)
        self._outputLayer.optimize(self._hiddenLayer.output(), learningRate)

