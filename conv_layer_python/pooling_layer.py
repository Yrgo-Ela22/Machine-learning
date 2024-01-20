import math
from enum import Enum
from conv_layer import newMatrix

class PoolType(Enum):
    Max = 0
    Average = 1

class PoolingLayer2D:

    # ------------------------------------------------------------------------------
    def __init__(self, size: int, type: PoolType = PoolType.Max) -> None:
        assert size > 0, "Invalid pooling layer size!"
        self._output = newMatrix(size, size)
        self._type = type

    # ------------------------------------------------------------------------------
    def output(self) -> tuple[list[float]]:
        return tuple(self._output)
    
    # ------------------------------------------------------------------------------
    def type(self) -> PoolType:
        return self._type
    
    # ------------------------------------------------------------------------------
    def size(self) -> int:
        return len(self._output)

    # ------------------------------------------------------------------------------
    def feedforward(self, input: tuple[tuple[float]]) -> None:
        assert len(input) > 0 and len(input[0]) > 0, "Invalid matrix dimensions!"
        assert len(input) > self.size() and len(input[0]) > self.size(), \
            "Pooling layer size must be less than the image to pool!"
        for i in range(self.size()):
            for j in range(self.size()):
                self._output[j][i] = self._pool(input, j, i)

    # ------------------------------------------------------------------------------
    def _numPaddings(self) -> int:
        return self.size() - 1
    
    # ------------------------------------------------------------------------------
    def _pool(self, input: tuple[tuple[float]], j: int, i: int) -> float:
        x = j * self.size()
        y = i * self.size()
        if not self._isWithinRange(input, x, y): return 0
        return self._poolMax(input, x, y) if self._type == PoolType.Max \
            else self._poolAverage(input, x, y)

    # ------------------------------------------------------------------------------
    def _poolMax(self, input: tuple[tuple[float]], x: int, y: int) -> float:
        maxVal = input[x][y]
        for i in range(self._iterationWidth(input)):
            for j in range(self._iterationHeight(input)):
                if x + i < len(input) and y + j < len(input[0]):
                    if input[x + i][y + j] > maxVal:
                        maxVal = input[x + i][y + j]
        return maxVal

     # ------------------------------------------------------------------------------
    def _poolAverage(self, input: tuple[tuple[float]], x: int, y: int) -> float:
        averageVal = 0.0
        for i in range(self._iterationWidth(input)):
            for j in range(self._iterationHeight(input)):
                averageVal += input[x + i][y + j]
        return averageVal / pow(self.size(), 2) 

    # ------------------------------------------------------------------------------
    def _iterationWidth(self, input: tuple[tuple[float]]) -> int:
        return int(len(input) / self.size() + 1) if self.size() < len(input) \
            else int(len(input) / self.size())
        
    # ------------------------------------------------------------------------------
    def _iterationHeight(self, input: tuple[tuple[float]]) -> int:
        return int(len(input[0]) / self.size() + 1) if self.size() < len(input[0]) \
            else int(len(input[0]) / self.size())
    
    # ------------------------------------------------------------------------------
    def _isWithinRange(self, input: tuple[tuple[float]], x: int, y: int) -> bool:
        return True if x < len(input) and y < len(input[0]) else False