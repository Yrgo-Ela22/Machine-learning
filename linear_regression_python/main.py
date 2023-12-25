from lin_reg import LinReg 

# ------------------------------------------------------------------------------
def main() -> None:
    trainInput = (0, 1, 2, 3, 4, 5)
    trainOutput = (-2, 1, 4, 7, 10)
    linReg = LinReg(trainInput, trainOutput)
    if linReg.train(1000, 0.01):
        linReg.printPredictions(trainInput)
    
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()