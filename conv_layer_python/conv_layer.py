
# ------------------------------------------------------------------------------
def newMatrix(width: int, height: int, startVal: float = 0.0) -> list[list[float]]:
    assert width >= 0 and height >= 0, "Invalid matrix dimensions!"
    matrix = []
    for i in range(width):
        row = []
        for j in range(height):
            row.append(startVal)
        matrix.append(row)
    return matrix

# ------------------------------------------------------------------------------
def pad(matrix: tuple[tuple[float]], numPaddings: int = 1) -> list[list[float]]:
    assert len(matrix) > 0 and len(matrix[0]) > 0, "Invalid matrix dimensions!"
    assert numPaddings >= 0, "Invalid number of paddings specified!"
    padded = newMatrix(len(matrix) + numPaddings * 2, len(matrix[0]) + numPaddings * 2)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            padded[i + numPaddings][j + numPaddings] = matrix[i][j]
    return padded

# ------------------------------------------------------------------------------
def roundMatrix(matrix: tuple[tuple[float]], numDecimals: int = 1) -> tuple[list[float]]:
    assert len(matrix) > 0 and len(matrix[0]) > 0, "Invalid matrix dimensions!"
    assert numDecimals >= 0, "Invalid number of decimals for rounding!"
    rounded = matrix
    for i in range(len(rounded)):
        for j in range(len(rounded[0])):
            rounded[i][j] = round(rounded[i][j], numDecimals)
    return tuple(rounded) 

# ------------------------------------------------------------------------------
def printMatrix(matrix: tuple[tuple[float]], numDecimals: int = 1, end: str = "\n") -> None:
    assert len(matrix) > 0 and len(matrix[0]) > 0, "Invalid matrix dimensions!"
    assert numDecimals >= 0, "Invalid number of decimals for rounding!"
    rounded = roundMatrix(matrix, numDecimals)
    print("--------------------------------------------------------------------------------", end="")
    for i in range(len(rounded)):
        print()
        for j in range(len(rounded[i])):
            print(rounded[j][i], end=" ")
    print("\n--------------------------------------------------------------------------------")

class ConvLayer2D:

    # ------------------------------------------------------------------------------
    def __init__(self, kernelSize: int) -> None:
        assert kernelSize > 0, "Kernel size must exceed 0!"
        self._inputPadded = []
        self._kernel = []
        self._output = []
        self._inputError = []
        self._kernelError = []
        self._initKernel(kernelSize)

    # ------------------------------------------------------------------------------
    def inputPadded(self) -> tuple[list[float]]:
        return tuple(self._inputPadded)
    
    # ------------------------------------------------------------------------------
    def kernel(self) -> tuple[list[float]]:
        return tuple(self._kernel)
    
     # ------------------------------------------------------------------------------
    def output(self) -> tuple[list[float]]:
        return tuple(self._output)
    
     # ------------------------------------------------------------------------------
    def inputError(self) -> tuple[list[float]]:
        return tuple(self._inputError)
    
     # ------------------------------------------------------------------------------
    def kernelError(self) -> tuple[list[float]]:
        return tuple(self._kernelError)
    
    # ------------------------------------------------------------------------------
    def imageWidth(self) -> int:
        return len(self._output)
    
    # ------------------------------------------------------------------------------
    def imageHeight(self) -> int:
        return len(self._output[0]) if len(self._output) > 0 else 0
    
    # ------------------------------------------------------------------------------
    def kernelSize(self) -> int:
        return len(self._kernel)
    
    # ------------------------------------------------------------------------------
    def numPaddings(self) -> int:
        return self.kernelSize() // 2
    
    # ------------------------------------------------------------------------------
    def feedforward(self, input: tuple[tuple[float]]) -> None:
        assert len(input) > 0, "Invalid size of input matrix!"
        self._inputPadded = self._pad(input)
        self._output = newMatrix(len(input), len(input[0]))
        for i in range(self.imageWidth()):
            for j in range(self.imageHeight()):
                for k in range(self.kernelSize()):
                    for l in range(self.kernelSize()):
                        self._output[i][j] += self._inputPadded[i + k][j + l] * self._kernel[k][l]
    
    # ------------------------------------------------------------------------------
    def backpropagate(self, outputError: tuple[tuple[float]]) -> None:
        assert len(outputError) >= 0 and len(outputError) == self.imageWidth() \
            and len(outputError[0]) == self.imageHeight(), "Invalid size of outputError matrix!"
        outputErrorPadded = self._pad(outputError)
        offset = self.kernelSize() - 1
        self._kernelError = newMatrix(self.kernelSize(), self.kernelSize())
        self._inputError = newMatrix(self.imageWidth(), self.imageHeight())
        for i in range(self.imageWidth()):
            for j in range(self.imageHeight()):
                for k in range(self.kernelSize()):
                    for l in range(self.kernelSize()):
                        self._kernelError[k][l] += self._inputPadded[i + k][j + l] * outputError[i][j]
                        self._inputError[i][j] += self._kernel[k][l] * \
                            outputErrorPadded[offset + i - k][offset + j - l]          
    
    # ------------------------------------------------------------------------------
    def optimize(self, learningRate: float = 0.01) -> None:
        assert learningRate > 0, "Learning rate must exceed 0!"
        for i in range(self.kernelSize()):
            for j in range(self.kernelSize()):
                self._kernel[i][j] += self._kernelError[i][j] * learningRate

    # ------------------------------------------------------------------------------
    def _initKernel(self, kernelSize: int) -> None:
        import random
        self._kernel = newMatrix(kernelSize, kernelSize)
        for i in range(kernelSize):
            for j in range(kernelSize):
                self._kernel[i][j] = random.random() * 10

    # ------------------------------------------------------------------------------
    def _pad(self, matrix: tuple[tuple[float]]) -> list[list[float]]:
        return pad(matrix, self.numPaddings())