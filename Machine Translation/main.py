import numpy as np
import pandas as pd
import sys
from RNN import RNN
from PreProcessing import dataPreprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    trainEnglish = "Data/english"
    trainFrench = "Data/french"
    with open(trainEnglish, "r") as f:
        fileData = f.read()
    engList = fileData.split('\n')
    with open(trainFrench, "r") as f:
        fileData = f.read()
    freList = fileData.split('\n')
    rawDF = {}
    rawDF["ENG"] = engList
    rawDF["FRENCH"] = freList
    df = pd.DataFrame(rawDF)
    dp_ins = dataPreprocessing()

    X_train, X_test, Y_train, Y_test, englishWordslength, frenchWordslength, countEnglishwords, countFrenchwords= dp_ins.preprocessingOfRawdata(
        df)


    r = RNN(englishWordslength, frenchWordslength, countEnglishwords, countFrenchwords,
                             dp_ins.wordsInput, dp_ins.encodedInput, dp_ins.wordsOutput,
                             dp_ins.encodedOutput)
    engTrain, engTest, freTrain, freTest = r.createDataSplit(engList,freList)

    phase = False
    if len(sys.argv) == 2:
        phase = (sys.argv[1].lower() == 'train')
    if phase:
        r.initializeRandomWeights(countFrenchwords)
        r.WA = np.load("Model Parameters/WA.npy").reshape(countFrenchwords, 1)
        r.WrnnA = np.load("Model Parameters/WrnnA.npy")
        r.WC = np.load("Model Parameters/WC.npy").reshape(countFrenchwords, 1)
        r.WrnnC = np.load("Model Parameters/WrnnC.npy")
        r.WB = np.load("Model Parameters/WB.npy").reshape(countFrenchwords, 1)
        r.WrnnB = np.load("Model Parameters/WrnnB.npy")
        r.DV = np.load("Model Parameters/DV.npy").reshape(countFrenchwords, 1)
        r.bA = np.load("Model Parameters/bA.npy")
        r.bB = np.load("Model Parameters/bB.npy")
        r.bC = np.load("Model Parameters/bC.npy")
        trainingError, testingError = r.functionofTrainingData(X_train, Y_train,10, X_test, Y_test)
        np.save("WA", r.WA)
        np.save("WrnnA", r.WrnnA)
        np.save("WC", r.WC)
        np.save("WrnnC", r.WrnnC)
        np.save("WB", r.WB)
        np.save("WrnnB", r.WrnnB)
        np.save("DV", r.DV)
        np.save("bA", r.bA)
        np.save("bB", r.bB)
        np.save("bC", r.bC)
        np.save("training error", testingError)
        np.save("testing error", testingError)
        print("model saved successfully..")
    else:
        r.testTranslation(engTest, freTest)