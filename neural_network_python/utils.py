import math
from enum import Enum
from numbers import Number

# ------------------------------------------------------------------------------
class ActFunc(Enum):
    Relu = 0,
    Tanh = 1

# ------------------------------------------------------------------------------
def relu(x: float) -> float:
    return x if x > 0.0 else 0.0

# ------------------------------------------------------------------------------
def reluDelta(x: float) -> float:
    return 1.0 if x > 0 else 0.0

# ------------------------------------------------------------------------------
def tanh(x: float) -> float:
    return math.tanh(x)

# ------------------------------------------------------------------------------
def tanhDelta(x: float) -> float:
    return 1 - math.pow(math.tanh(x), 2)

# ------------------------------------------------------------------------------
def actFuncOutput(x: float, actFunc: ActFunc) -> float:
    return relu(x) if actFunc == ActFunc.Relu else tanh(x)

# ------------------------------------------------------------------------------
def actFuncDelta(x: float, actFunc: ActFunc) -> float:
    return reluDelta(x) if actFunc == ActFunc.Relu else tanhDelta(x)

# ------------------------------------------------------------------------------
def initList(list: list[int], numValues: int, startVal: Number) -> None:
    list.clear()
    for i in range(numValues):
        list.append(startVal)

# ------------------------------------------------------------------------------
def initRandomList(list: list[int], numValues: int) -> None:
    list.clear()
    for i in range(numValues):
        list.append(getRandomStartVal())

# ------------------------------------------------------------------------------
def initRandom2DList(list: list[list[int]], numColumns: int, numRows: int) -> None:
    list.clear()
    for i in range(numColumns):
        column = []
        for j in range(numRows):
            column.append(getRandomStartVal())
        list.append(column)

# ------------------------------------------------------------------------------
def getRandomStartVal() -> float:
    import random
    return random.random()

# ------------------------------------------------------------------------------
min = (lambda x, y: x if x < y else y)

# ------------------------------------------------------------------------------
max = (lambda x, y: x if x > y else y)

# ------------------------------------------------------------------------------
def toContainer(arg) -> list | tuple:
    return arg if type(arg) is list or type(arg) is tuple else [arg]
    
# ------------------------------------------------------------------------------
def printFloats(data: tuple[float], 
                numDecimals: int = 1, 
                space: str = " ", 
                end: str = "\n") -> None:
    for i in data:
        print(round(i, numDecimals), end = space)
    print(end, end="")