import string
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import re


class dataPreprocessing:
    def __init__(self):
        print("reading the data.....")

    def preprocessingOfRawdata(self, df):
        df.drop_duplicates(subset=None, keep='first')
        df = shuffle(df)
        df['ENG'] = df['ENG'].apply(lambda x: x.lower())
        df['FRENCH'] = df['FRENCH'].apply(lambda x: x.lower())
        df['ENG'] = df['ENG'].apply(lambda x: re.sub("'", '', x))
        df['FRENCH'] = df['FRENCH'].apply(lambda x: re.sub("'", '', x))
        removalOfPunc = set(string.punctuation)
        df['ENG'] = df['ENG'].apply(lambda x: ''.join(ch for ch in x if ch not in removalOfPunc))
        df['FRENCH'] = df['FRENCH'].apply(lambda x: ''.join(ch for ch in x if ch not in removalOfPunc))
        df['ENG'] = df['ENG'].apply(lambda x: x.strip())
        df['FRENCH'] = df['FRENCH'].apply(lambda x: x.strip())

        df['ENG'] = df['ENG'].apply(lambda x: re.sub(" +", " ", x))
        df['FRENCH'] = df['FRENCH'].apply(lambda x: re.sub(" +", " ", x))

        englishWordslength = 0
        uniqueEnglishwords = set()
        for engWords in df['ENG']:
            engWordslist = engWords.split()
            if englishWordslength < len(engWordslist):
                englishWordslength = len(engWordslist)
            for phrase in engWordslist:
                if phrase not in uniqueEnglishwords:
                    uniqueEnglishwords.add(phrase)

        self.englishWordslength = englishWordslength

        frenchWordslength = 0
        uniqueFrenchwords = set()
        for frenchWords in df['FRENCH']:
            frenchWordslist = frenchWords.split()
            if frenchWordslength  < len(frenchWordslist):
                frenchWordslength = len(frenchWordslist)
            for phrase in frenchWords.split():
                if phrase not in uniqueFrenchwords:
                    uniqueFrenchwords.add(phrase)
        self.frenchWordslength  = frenchWordslength

        wordsInput = sorted(list(uniqueEnglishwords))
        wordsOutput = sorted(list(uniqueFrenchwords))
        self.wordsInput = wordsInput
        self.wordsOutput = wordsOutput

        inputOfEncoder = LabelEncoder()
        indexOfinput = inputOfEncoder.fit_transform(wordsInput)
        indexOfoutput = inputOfEncoder.fit_transform(wordsOutput)

        funcofOnehotencoder = OneHotEncoder(sparse=False)
        indexOfinput = indexOfinput.reshape(len(indexOfinput), 1)
        self.encodedInput = funcofOnehotencoder.fit_transform(indexOfinput)
        indexOfoutput = indexOfoutput.reshape(len(indexOfoutput), 1)
        self.encodedOutput = funcofOnehotencoder.fit_transform(indexOfoutput)

        if (len(wordsInput) < len(wordsOutput)):
            diffOfinputandoutput = len(wordsOutput) - len(wordsInput)
            paddingOfZeroes = np.zeros((1, len(wordsInput)))
            for i in range(diffOfinputandoutput):
                self.encodedInput = np.concatenate((self.encodedInput, paddingOfZeroes.T), axis=1)
        else:
            diffOfinputandoutput = len(wordsInput) - len(wordsOutput)
            paddingOfZeroes = np.zeros((1, len(wordsOutput)))
            for i in range(diffOfinputandoutput):
                self.encodedOutput = np.concatenate((self.encodedOutput, paddingOfZeroes.T), axis=1)

        columnofEngdata = df['ENG'].to_numpy()
        columnofFrenchdata = df['FRENCH'].to_numpy()

        index = np.random.permutation(columnofEngdata.shape[0])

        trainingIndex, testingIndex = index[:int((columnofEngdata.shape[0] * 0.8))], index[
                                                                                int((columnofEngdata.shape[0] * 0.8)):]
        X_train, X_test = columnofEngdata[trainingIndex], columnofEngdata[testingIndex]
        Y_train, Y_test = columnofFrenchdata[trainingIndex], columnofFrenchdata[testingIndex]

        return X_train, X_test, Y_train, Y_test, englishWordslength, frenchWordslength, len(uniqueEnglishwords), len(uniqueFrenchwords)
