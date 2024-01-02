
class FlattenLayer:

    # ------------------------------------------------------------------------------
    def __init__(self) -> None:
        self._output = []

    # ------------------------------------------------------------------------------
    def output(self) -> tuple[float]:
        return tuple(self._output)
    
    # ------------------------------------------------------------------------------
    def size(self) -> int:
        return len(self._output)
    
    # ------------------------------------------------------------------------------
    def feedforward(self, input: tuple[tuple[float]]) -> None:
        assert len(input) > 0 and len(input[0]) > 0, "Invalid matrix dimensions!"
        self._output.clear()
        for row in input:
            for value in row:
                self._output.append(value)

            
    # ------------------------------------------------------------------------------
    def print(self, numDecimals: int = 1) -> None:
        assert numDecimals >= 0, "Invalid number of decimals!"
        if self.size() == 0: return
        print("--------------------------------------------------------------------------------")
        for value in self._output:
            print(round(value, numDecimals), end = " ")
        print("\n--------------------------------------------------------------------------------\n")


