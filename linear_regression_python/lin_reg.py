import random 

class LinReg:

    # ------------------------------------------------------------------------------
    def __init__(self, trainInput: list[float], trainOutput: list[float]) -> None:
        self._trainInput = self._toContainer(trainInput)
        self._trainOutput = self._toContainer(trainOutput)
        self._trainOrder = []
        self._bias = self._getRandomStartVal()
        self._weight = self._getRandomStartVal()
        self._checkTrainingData()
        self._initTrainOrderList()
    
    # ------------------------------------------------------------------------------
    def bias(self) -> float:
        return self._bias

    # ------------------------------------------------------------------------------
    def weight(self) -> float:
        return self._weight
    
    # ------------------------------------------------------------------------------
    def numTrainingSets(self) -> int:
        return len(self._trainOrder)

    # ------------------------------------------------------------------------------
    def train(self, numEpochs: int, learningRate: float = 0.01) -> bool:
        if self.numTrainingSets() == 0 or numEpochs <= 0 or learningRate <= 0:
            return False
        for i in range(numEpochs):
            self._randomizeTrainingOrder()
            for j in self._trainOrder:
                self._optimize(self._trainInput[j], self._trainOutput[j], learningRate)
        return True

    # ------------------------------------------------------------------------------
    def predict(self, input: float) -> float:
        return self._weight * input + self._bias    
    
    # ------------------------------------------------------------------------------
    def printPredictions(self, inputs: list[float], numDecimals: int = 1) -> None:
        if len(inputs) == 0 or numDecimals < 0: return
        print("--------------------------------------------------------------------------------")
        for input in inputs:
            print("Input:\t" + str(round(input, numDecimals)))
            print("Output:\t" + str(round(self.predict(input))))
            if input != inputs[len(inputs) - 1]: print()
        print("--------------------------------------------------------------------------------\n")

    # ------------------------------------------------------------------------------
    def _checkTrainingData(self) -> None:
        if len(self._trainInput) != len(self._trainOutput):
            numSets = min(len(self._trainInput), len(self._trainOutput))
            self._trainInput = self._trainInput[:numSets]
            self._trainOutput = self._trainOutput[:numSets]

    # ------------------------------------------------------------------------------
    def _initTrainOrderList(self) -> None:
        for i in range(len(self._trainInput)):
            self._trainOrder.append(i)

    # ------------------------------------------------------------------------------
    def _randomizeTrainingOrder(self) -> None:
        for i in range(self.numTrainingSets()):
            r = random.randint(0, self.numTrainingSets() - 1)
            temp = self._trainOrder[i]
            self._trainOrder[i] = self._trainOrder[r]
            self._trainOrder[r] = temp     

    # ------------------------------------------------------------------------------
    def _optimize(self, input: float, output: float, learningRate: float) -> None:
        if input != 0:
            error = output - self.predict(input)
            self._bias += error * learningRate
            self._weight += error * learningRate * input
        else:
            self._bias = output
        
    # ------------------------------------------------------------------------------
    @staticmethod
    def _toContainer(arg) -> list | tuple:
        return arg if type(arg) == list or type(arg) == tuple else [arg]
    
    # ------------------------------------------------------------------------------
    @staticmethod
    def _getRandomStartVal() -> float:
        return random.random()
    
    # ------------------------------------------------------------------------------
    @staticmethod
    def _printFloats(data: tuple[float], 
                     numDecimals: int = 1, 
                     space: str = " ", 
                     end: str = "\n") -> None:
        for i in data:
            print(round(i, numDecimals), end=space)
        print(end, end="")
    
    # ------------------------------------------------------------------------------
    min = staticmethod(lambda x, y: x if x < y else y)