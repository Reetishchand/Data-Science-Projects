import heapq
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from rouge_score import rouge_scorer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
np.random.seed(1)
class RNN:
    def __init__(self, englishWordslength, frenchWordslength, countEnglishwords, countFrenchwords, wordsInput,
                 encodedInput, wordsOutput, encodedOutput):
        self.countFrenchwords, self.encodedInput = countFrenchwords, encodedInput
        self.wordsInput ,self.englishWordslength,self.frenchWordslength = wordsInput,englishWordslength,frenchWordslength
        self.encodedOutput,self.countEnglishwords,self.wordsOutput = encodedOutput,countEnglishwords,wordsOutput
        self.bC,self.WrnnB,self.bA,self.WrnnA,self.bB,self.WrnnC = np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1)
        if  self.countEnglishwords < self.countFrenchwords:
            maximumLength = self.countFrenchwords
        else:
            maximumLength = self.countEnglishwords
        self.maximumLength = maximumLength
        self.WB ,self.WA,self.DV,self.WC = np.random.rand(maximumLength, 1),np.random.rand(maximumLength, 1), np.random.rand(maximumLength, 1),np.random.rand(maximumLength, 1)
        print("Initializion of parameters done....")

    def tanhActivationfunction(self, x):
        """
        The tanh_activation_func applies tanh function to the input and returns the result value
        parameters are x: input value
        returns tanh(x)
        """
        return (2 * self.sigmoidActivationfunction(2 * x)) - 1

    def sigmoidActivationfunction(self, sig):
        """
        The sigmoid_activation_func applies sigmoid function and returns the result value
        parameters are sig: input value
        returns sigmoid(x)
        """
        sigmoidValue = 1 / (1 + np.exp(-sig, dtype=np.float64))
        sigmoidValue = np.minimum(sigmoidValue, 0.9999)
        sigmoidValue = np.maximum(sigmoidValue, 0.0001)
        return sigmoidValue

    def softmaxActivationfunction(self, val):
        """
        The softmax_activation_func applies softmax function to the input and returns the result value
        parameters are  val: input value
        returns softmax(val)
        """
        outputVal = self.DV * val
        outputVal[0] = 0
        outputVal[1] = 0
        tep=outputVal - np.max(outputVal)
        finalOutput = np.exp(tep)
        return np.transpose(finalOutput/finalOutput.sum())

    def tanhDerivativefunction(self, net_val):
        """
        The derivative_of_tanh_func applies calculation taking net_val as input and returns the result value
        parameters are net_val:input value
        returns derivative_output
        """
        return (1 - np.square(np.tanh(net_val)))

    def findPositionofelement(self, arrayOfelements):
        """
        the pos_find function returns index of element  having value 1 in the array
        parameters are arr which is an array
        returns the index if satisfies the condition otherwise zero
        """
        for i in range(len(arrayOfelements)):
            if arrayOfelements[i] == 1.0:
                return i
        return 0

    def crossEntropyfunction(self, predVal, outVal, ep=1e-12):
        """
        The cross_entropy_func function calculates the cross entropy value for the given pred_val and output_val
        parameters are  pred_val which is predicted output, output_val which is expected output and ep which is the epsilon value.
        returns the calculated cross entropy value
        """
        predVal = np.clip(predVal, ep, 1. - ep)
        K = self.findPositionofelement(outVal)
        I = predVal.shape[0]
        crossEntropy = -np.sum(outVal[K] * np.log(outVal[K] + 1e-9))
        crossEntropy = crossEntropy/ I
        return crossEntropy

    def totalCrossEntropyfunction(self, predVal, outputVal):
        """
        The total_cross_ent_func function calculates the total cross entropy for entire sentence
        parameters are pred_val which is the array of predicted values, output_val which is  array of expected output values
        returns the calculated total cross entropy
        """
        errorValue = 0
        M = self.frenchWordslength
        for n in range(M - 1):
            retVal = self.crossEntropyfunction(predVal[n], outputVal[n + 1])
            errorValue = errorValue + retVal
        return errorValue


    def embeddedLayerfunction(self, X, TypeOfRNN):
        """
        The embedded_layer_func function returns array of encoded vectors for the provided input X
        parameters are the Input sentence X and Typ_of_RNN need to encode or decode the sentence
        returns Array of  encoded vectors for the provided input sentences X
        """
        arrayOfencoder,countOfwords =  [],0
        if "encoder"==TypeOfRNN :
            lengthOfenglish = self.encodedInput.shape[1]
            paddingofZeroes,sentenceOfwords = [0] * lengthOfenglish, X.split()
            for each in sentenceOfwords:
                x1  = self.wordsInput.index(each)
                x2=self.encodedInput[x1]
                arrayOfencoder.append(x2)
                countOfwords +=1

            while  self.englishWordslength > countOfwords:
                arrayOfencoder.append(paddingofZeroes)
                countOfwords += 1
        else:
            lengthOffrench = self.encodedOutput.shape[1]
            paddingofZeroes = [0] * lengthOffrench
            sentenceOfwords = X.split()
            for each in sentenceOfwords:
                arrayOfencoder.append(self.encodedOutput[self.wordsOutput.index(each)])
                countOfwords = countOfwords + 1

            while countOfwords < self.frenchWordslength:
                arrayOfencoder.append(paddingofZeroes)
                countOfwords += 1
        encodeNPArr  =np.array(arrayOfencoder)
        return encodeNPArr


    def forwardPassfunction(self, X, TypeOfRNN, IntialState,HiddenState):
        """
        The function forwardPass_func performs the forward pass through RNN
        parameters are Input sentence X, Type of RNN ie encoder, train_decoder or test_decoder,
        initial state which is previous internal state and hidden state which is previous hidden state
        returns  initial state, current hidden state and output value
        """
        encoderValue = self.embeddedLayerfunction(X, TypeOfRNN)
        global initialCellstate, hiddenOutputstate, outputArray


        if "test_decoder"==TypeOfRNN:

            N = self.frenchWordslength
            outputArray,initialCellstate,hiddenOutputstate = np.zeros((N, self.maximumLength)), np.zeros((N, 1)),np.zeros((N, 1))
            outputArray[0][0] = 1
            A = self.tanhActivationfunction(np.dot(outputArray[0], self.WA) + np.dot(self.WrnnA, HiddenState) + self.bA)
            B = self.sigmoidActivationfunction(np.dot(outputArray[0], self.WB) + np.dot(self.WrnnB, HiddenState) + self.bB)
            C = self.sigmoidActivationfunction(np.dot(outputArray[0], self.WC) + np.dot(self.WrnnC, HiddenState) + self.bC)
            initialCellstate[0] = A * B + IntialState
            hiddenOutputstate[0] = self.tanhActivationfunction(initialCellstate[0]) * C
            outputArray[0] = self.softmaxActivationfunction(hiddenOutputstate[0])

            for iterator in np.arange(1, N - 1):
                A = self.tanhActivationfunction(
                    np.dot(outputArray[iterator-1], self.WA) + np.dot(self.WrnnA, hiddenOutputstate[iterator - 1]) + self.bA)
                B = self.sigmoidActivationfunction(
                    np.dot(outputArray[iterator-1], self.WB) + np.dot(self.WrnnB, hiddenOutputstate[iterator - 1]) + self.bB)
                C = self.sigmoidActivationfunction(
                    np.dot(outputArray[iterator-1], self.WC) + np.dot(self.WrnnC, hiddenOutputstate[iterator - 1]) + self.bC)
                initialCellstate[iterator] = A * B + initialCellstate[iterator - 1]
                hiddenOutputstate[iterator] = self.tanhActivationfunction(initialCellstate[iterator]) * C
                outputArray[iterator] = self.softmaxActivationfunction(hiddenOutputstate[iterator])

        elif "encoder"==TypeOfRNN:
            N = self.englishWordslength
            initialCellstate,hiddenOutputstate,outputArray = np.zeros((N, 1)),np.zeros((N, 1)),np.zeros((N - 1, 1))

            for iterator in np.arange(N):
                A = self.tanhActivationfunction(
                    np.dot(encoderValue[iterator], self.WA) + np.dot(self.WrnnA, hiddenOutputstate[iterator - 1]) + self.bA)
                B = self.sigmoidActivationfunction(
                    np.dot(encoderValue[iterator], self.WB) + np.dot(self.WrnnB, hiddenOutputstate[iterator - 1]) + self.bB)
                C = self.sigmoidActivationfunction(
                    np.dot(encoderValue[iterator], self.WC) + np.dot(self.WrnnC, hiddenOutputstate[iterator - 1]) + self.bC)
                initialCellstate[iterator] = A * B + initialCellstate[iterator - 1]
                hiddenOutputstate[iterator] = self.tanhActivationfunction(initialCellstate[iterator]) * C

        elif "train_decoder"==TypeOfRNN :
            N = self.frenchWordslength
            hiddenOutputstate ,initialCellstate,outputArray = np.zeros((N, 1)),np.zeros((N, 1)),np.zeros((N, self.maximumLength))
            A = self.tanhActivationfunction(np.dot(encoderValue[0], self.WA) + np.dot(self.WrnnA, HiddenState) + self.bA)
            B = self.sigmoidActivationfunction(np.dot(encoderValue[0], self.WB) + np.dot(self.WrnnB, HiddenState) + self.bB)
            C = self.sigmoidActivationfunction(np.dot(encoderValue[0], self.WC) + np.dot(self.WrnnC, HiddenState) + self.bC)
            initialCellstate[0] = A * B + IntialState
            hiddenOutputstate[0] = self.tanhActivationfunction(initialCellstate[0]) * C
            outputArray[0] = self.softmaxActivationfunction(hiddenOutputstate[0])
            for iterator in np.arange(1, N - 1):
                A = self.tanhActivationfunction(
                    np.dot(encoderValue[iterator], self.WA) + np.dot(self.WrnnA, hiddenOutputstate[iterator - 1]) + self.bA)
                B = self.sigmoidActivationfunction(
                    np.dot(encoderValue[iterator], self.WB) + np.dot(self.WrnnB, hiddenOutputstate[iterator - 1]) + self.bB)
                C = self.sigmoidActivationfunction(
                    np.dot(encoderValue[iterator], self.WC) + np.dot(self.WrnnC, hiddenOutputstate[iterator - 1]) + self.bC)
                initialCellstate[iterator] = A * B + initialCellstate[iterator - 1]
                hiddenOutputstate[iterator] = self.tanhActivationfunction(initialCellstate[iterator]) * C
                outputArray[iterator] = self.softmaxActivationfunction(hiddenOutputstate[iterator])
                if np.any(encoderValue[iterator + 1]):
                    self.functionForBackwardpass(A, B, C, initialCellstate[iterator], hiddenOutputstate[iterator], encoderValue[iterator], outputArray[iterator],
                                           encoderValue[iterator + 1])


        return initialCellstate, hiddenOutputstate, outputArray

    def computingDeltaweights(self, hiddenState, targetValue, predictedValue):
        """
        The delta weight function calculates the derivative for output softmax layer
        parameters are hidden state , target output and the predicted output
        returns  derivative of error
        """
        derivativeOfoutput = np.zeros(self.DV.shape)
        dervative = predictedValue - targetValue
        return dervative * hiddenState

    def functionForBackwardpass(self, A, B, C, initialState, hiddenState, presentValueofX, presentValueofY, expectedValueofY,
                          learningRate=0.01):
        errorOfsoftmax,softmaxDeltaoutput = presentValueofY - expectedValueofY,self.computingDeltaweights(hiddenState, expectedValueofY, presentValueofY)
        derivativeOfInitialstate = self.tanhDerivativefunction(initialState)
        intermediateValueofWA = (C * derivativeOfInitialstate * B * (1 - np.square(A)))
        derivativeofWA,derivativeofWrnnA = intermediateValueofWA * presentValueofX, intermediateValueofWA * hiddenState
        deltaValueofWrnnA = np.sum(errorOfsoftmax * derivativeofWrnnA)
        deltaValueofWa = errorOfsoftmax * derivativeofWA
        intermediateValueofWB = C * derivativeOfInitialstate * A * (B * (1 - B))
        derivativeofWB = intermediateValueofWB * presentValueofX
        deltaValueofWB,derivativeofWrnnB = errorOfsoftmax * derivativeofWB,intermediateValueofWB * hiddenState
        deltaValueofWrnnB = np.sum(errorOfsoftmax * derivativeofWrnnB)
        intermediateValueofWC = np.tanh(initialState) * C * (1 - C)
        derivativeofWC = intermediateValueofWC * presentValueofX
        derivativeofWrnnC = intermediateValueofWC * hiddenState
        deltaValueofWrnnC = np.sum(errorOfsoftmax * derivativeofWrnnC)
        deltaValueofWC = errorOfsoftmax * derivativeofWC
        self.WB = self.WB - (learningRate * deltaValueofWB).reshape(self.maximumLength, 1)
        self.WrnnC = self.WrnnC - (learningRate * deltaValueofWrnnC)
        self.WA = self.WA - (learningRate * deltaValueofWa).reshape(self.maximumLength , 1)
        self.WC = self.WC - (learningRate * deltaValueofWC).reshape(self.maximumLength , 1)
        self.WrnnA = self.WrnnA - (learningRate * deltaValueofWrnnA)
        self.DV = self.DV - (learningRate * softmaxDeltaoutput).reshape(self.maximumLength , 1)
        self.WrnnB = self.WrnnB - (learningRate * deltaValueofWrnnB)

    def functionofTrainingData(self, X_train, Y_train , epoch, X_test, Y_test):

        testingError,trainingError = [],[]
        for iterator in range(epoch):
            print(" Running Epoch :", iterator)
            totalErroroftrainingnetwork = 0
            for i in range(X_train.size):
                var = 1
                if (len(X_train[i].split()) < 3):
                    var = 10
                for each in range(var):
                    EncoderInitialstate, EncoderHiddenState, EncoderOutput = self.forwardPassfunction(X_train[i], "encoder",
                                                                                              0, 0)
                    DTrainInitialstate, DTrainHiddenState, DTrainOutput = self.forwardPassfunction(Y_train[i],
                                                                                               "train_decoder",
                                                                                               EncoderInitialstate[
                                                                                                   EncoderInitialstate.shape[
                                                                                                       0] - 1],
                                                                                               EncoderHiddenState[
                                                                                                   EncoderHiddenState.shape[
                                                                                                       0] - 1])

                    outputOfEncoder = self.embeddedLayerfunction(Y_train[i], "decoder")
                    ErrorafterapplyingEntropy = self.totalCrossEntropyfunction(DTrainOutput, outputOfEncoder)
                    totalErroroftrainingnetwork = totalErroroftrainingnetwork + ErrorafterapplyingEntropy

            trainingError.append(totalErroroftrainingnetwork / X_train.size)
            testingError.append(totalErroroftrainingnetwork / X_train.size)
            print("Completed Epoch :",iterator)
        return trainingError, testingError


    def initializeTestFunction(self):
        trainEnglish = "Data/english"
        trainFrench = "Data/french"
        with open(trainEnglish, "r") as f:
            fileData = f.read()
        englishSentences = fileData.split('\n')
        with open(trainFrench, "r") as f:
            fileData = f.read()
        frenchSentences = fileData.split('\n')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(englishSentences)
        preProcEng, engTokens = tokenizer.texts_to_sequences(englishSentences), tokenizer
        preProcEng = pad_sequences(preProcEng, maxlen=None, padding='post')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(frenchSentences)
        preProcFre, freTokens = tokenizer.texts_to_sequences(frenchSentences), tokenizer
        preProcFre = pad_sequences(preProcFre, maxlen=None, padding='post')
        preProcFre = preProcFre.reshape(*preProcFre.shape, 1)
        model = keras.models.load_model('Model Parameters')
        print("model loaded....")
        return freTokens,preProcEng,engTokens,model

    def translateSentence(self, sentence,freTokens,preProcEng,engTokens,model):
        try:
            y_id_to_word = {value: key for key, value in freTokens.word_index.items()}
            y_id_to_word[0] = ''
            sentence = re.sub(r'[^\w\s]', '', sentence).lower()
            sentence = [engTokens.word_index[word] for word in sentence.split()]
            sentence = pad_sequences([sentence], maxlen=preProcEng.shape[-1], padding='post')
            sentences = np.array([sentence[0], preProcEng[0]])
            predictions = model.predict(sentences, len(sentences))
            mypred = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
            return mypred+" ."
        except:
            return ""

    def testTranslation(self, englishSentences, frenchSentences):
        freTokens,preProcEng,engTokens,model=self.initializeTestFunction()
        hypothesisList = []
        actualList = []
        for i in range(len(englishSentences)):
            print(i + 1)
            sentence = englishSentences[i]
            print("Query :", sentence)
            hypothesis = self.translateSentence(sentence,freTokens,preProcEng,engTokens,model)
            print("Predicted : ", hypothesis)
            hypothesisList.append(hypothesis)
            actualList.append(frenchSentences[i])
            print("Actual :", frenchSentences[i])
        self.viewMetrics(hypothesisList, actualList)

    def TranslationfromEngtoFrench(self,engInputsentence):
        EncoderInitialstate, EncoderHiddenstate, EncoderOutput = self.forwardPassfunction(engInputsentence, "encoder", 0,
                                                                                  0)
        N = self.frenchWordslength
        outputArray,hiddenOutputstate,initialCellstate = np.zeros((N, self.maximumLength )),np.zeros((N, 1)),np.zeros((N, 1))
        A = self.tanhActivationfunction(np.dot(outputArray[0], self.WA) + np.dot(self.WrnnA, EncoderHiddenstate[EncoderHiddenstate.shape[0] - 1]) + self.bA)
        B = self.sigmoidActivationfunction(np.dot(outputArray[0], self.WB) + np.dot(self.WrnnB, EncoderHiddenstate[EncoderHiddenstate.shape[0] - 1]) + self.bB)
        C = self.sigmoidActivationfunction(np.dot(outputArray[0], self.WC) + np.dot(self.WrnnC, EncoderHiddenstate[EncoderHiddenstate.shape[0] - 1]) + self.bC)
        initialCellstate[0] = A * B + EncoderInitialstate[EncoderInitialstate.shape[0] - 1]
        hiddenOutputstate[0] = self.tanhActivationfunction(initialCellstate[0]) * C
        outputArray[0] = self.softmaxActivationfunction(hiddenOutputstate[0])

        for each in np.arange(1, N - 1):
            A = self.tanhActivationfunction(
                np.dot(outputArray[each - 1], self.WA) + np.dot(self.WrnnA, hiddenOutputstate[each - 1]) + self.bA)
            B = self.sigmoidActivationfunction(
                np.dot(outputArray[each - 1], self.WB) + np.dot(self.WrnnB, hiddenOutputstate[each - 1]) + self.bB)
            C = self.sigmoidActivationfunction(
                np.dot(outputArray[each - 1], self.WC) + np.dot(self.WrnnC, hiddenOutputstate[each - 1]) + self.bC)
            initialCellstate[each] = A * B + initialCellstate[each - 1]
            hiddenOutputstate[each] = self.sigmoidActivationfunction(initialCellstate[each]) * C
            outputArray[each] = self.softmaxActivationfunction(hiddenOutputstate[each])
        translationOutput = self.convertingTosentences(outputArray)
        print(translationOutput)

    def initializeRandomWeights(self,size):
        arr=[]
        for i in range(size):
            arr.append(random.random())
        arr=np.array(arr)
        np.save('Model Parameters/WA.npy', arr)
        np.save('Model Parameters/WC.npy', arr)
        np.save('Model Parameters/WB.npy', arr)
        np.save('Model Parameters/DV.npy', arr)

    def createDataSplit(self,englishSentences,frenchSentences):

        s = set([])
        i = 0
        target = 500
        while i < target:
            q = random.randint(1, len(englishSentences) - 1)
            if q not in s:
                s.add(q)
                i += 1
        engTrain, engTest, freTrain, freTest = [], [], [], []
        for i in range(len(englishSentences)):
            if i in s:
                engTest.append(englishSentences[i])
                freTest.append(frenchSentences[i])
            else:
                engTrain.append(englishSentences[i])
                freTrain.append(frenchSentences[i])
        return engTrain, engTest, freTrain, freTest

    def convertingTosentences(self, targetOutput):
        try:
            N = self.frenchWordslength
            expectedSentence = ""
            for i in range(N - 1):
                indexOfprobability = np.argmax(targetOutput[i])
                indices = heapq.nlargest(3, range(len(targetOutput[i])), targetOutput[i].__getitem__)
                expectedSentence = expectedSentence + " " + self.wordsOutput, [indexOfprobability]
            return expectedSentence
        except:
            return ""


    def viewMetrics(self, hypothesis, prediction):
        p1, p2, fm1, fm2, r1, r2, rL, pL, fmL = [], [], [], [], [], [], [], [], []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL', "rouge2"], use_stemmer=True)
        for i in range(len(hypothesis)):
            scores = scorer.score(hypothesis[i], prediction[i])
            p1.append(scores["rouge1"].precision)
            fm1.append(scores["rouge1"].fmeasure)
            r1.append(scores["rouge1"].recall)
            pL.append(scores["rougeL"].precision)
            fmL.append(scores["rougeL"].fmeasure)
            rL.append(scores["rougeL"].recall)
            p2.append(scores["rouge2"].precision)
            fm2.append(scores["rouge2"].fmeasure)
            r2.append(scores["rouge2"].recall)
        print("==================================================================================")
        print("ROUGE-1 METRICS ")
        print("fmeasure - ", sum(fm1) / len(fm1))
        print("precision - ", sum(p1) / len(p1))
        print("recall - ", sum(r1) / len(r1))
        print("==================================================================================")
        print("ROUGE-2 METRICS ")
        print("fmeasure - ", sum(fm2) / len(fm2))
        print("precision - ", sum(p2) / len(p2))
        print("recall - ", sum(r2) / len(r2))
        print("==================================================================================")
        print("ROUGE-L METRICS ")
        print("fmeasure - ", sum(fm2) / len(fm2))
        print("precision - ", sum(p2) / len(p2))
        print("recall - ", sum(r2) / len(r2))
        print("==================================================================================")

