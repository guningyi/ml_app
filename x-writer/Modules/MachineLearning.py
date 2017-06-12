import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from Modules.ConsoleOutput import ConsoleOutput
from Modules.NaturalLanguage import NaturalLanguageObject

_MAX_DECIMAL_PLACES = 10

# This svm attempts to predict the next identifier in a sequence
# And what comes next
class NNSentenceStructure:
    trainingData = []
    trainingDataResults = []
    # clf是分类器的缩写，classifier
    clf = None

    # adds data to the training buffer
    def loadVectorsIntoNetwork(self, inNormalisedData, targetResult):
        self.trainingData.extend(inNormalisedData) # 在已经存在的列表尾部一次性的追加多个元素
        self.trainingDataResults.extend(targetResult)

    # Fits the network to all of the data passed in
    def FitNetwork(self):
        countItems = len(self.trainingDataResults)

        self._fit(self.trainingData, self.trainingDataResults)

        ConsoleOutput.printGreen("Data successfully fitted to the sentence structure network.")
        ConsoleOutput.printGreen("Vectors: " + str(countItems))

        self.trainingData = None
        self.trainingDataResults = None


    def _fit(self, dataVector, targetVector):
        '''
        print('\n pf:')
        print(dataVector)
        print(targetVector)
        print('\n')
        '''
        #asarray()，功能是将输入数据转化为数组
        self.clf.fit(np.asarray(dataVector, dtype="float"), np.asarray(targetVector, dtype="float"))

    # gets a prediction from the network with the given input
    def getPrediction(self, inNormalisedData):
        #print(inNormalisedData)
        pred = self.clf.predict(inNormalisedData)
        return float(round(pred[0], _MAX_DECIMAL_PLACES))

    def getPredictionProbability(self, inNormalisedData):
        predProb = self.clf.predict_proba(inNormalisedData)
        return predProb


    def __init__(self):
        #self.clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
        # kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        #self.clf = linear_model.SGDClassifier()

        # 26% accuracy
        self.clf = KNeighborsClassifier() #初始化的时候，为clf赋值 KNeighborsClassifier(),而这个KNeighborsClassifier是从sklearn.neighbors import进来的。
                                          #实际上就是一个K近邻算法的构造器
        # Bad accuracy (1%) but makes sense
        #self.clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.0,
        #kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)



class NNVocabulary:
    trainingData = []
    trainingDataResults = []
    clf = None
    _Networks = []
    _Vocabulary = None

    # adds data to the training buffer
    def loadVectorsIntoNetworkByIndex(self, index, inNormalisedData, targetResult):
        self.trainingData[index].append([inNormalisedData])
        self.trainingDataResults[index].append(targetResult)

    # Loads the vacabulary data localy into the machine learning object.
    # It matches the normal to the result ex.. ([0.1332112], 'hello')
    # if [0.1332112] is passed in then it will be matched to the word 'hello'
    def loadVocab(self, index, inNormal, inResult):
        self._Vocabulary[index].append((inNormal, inResult))
    # Uses the passed in vocabulary to convert the given normal back into a word
    def _getFromVocab(self, inIdentifier, inNormal):
        for index, i in enumerate(NaturalLanguageObject._Identifiers):
            # get the index
            if(inIdentifier == i):
                # locate normal
                for index2, val in enumerate(self._Vocabulary[index]):
                    if(val[0] == inNormal):
                        return val[1]

    # Fits the network to all of the data passed in
    def FitNetwork(self):
        countItems = 0
        # print('size of trainingData[6] is:')
        # print(len(self.trainingData[6]))
        # print(self.trainingData[6])
        # train all of the networks at once
        # trainingData[] 是二维数组,
        # len(trainingData) 是　45
        # trainingData[6]的size是2233
        for index, val in enumerate(self.trainingData):
            if(len(self.trainingData[index]) > 0):
                self._fit(index, self.trainingData[index], self.trainingDataResults[index])
                countItems = countItems + len(self.trainingData[index])
            else:
                ConsoleOutput.printRed("No training data for vocab identifier: " + NaturalLanguageObject._Identifiers[index])

        ConsoleOutput.printGreen("Data successfully fitted to the vocabulary network.")
        ConsoleOutput.printGreen("Vectors: " + str(countItems))
        print("\n")

        self.trainingData = None
        self.trainingDataResults = None

    def _fit(self, index, dataVector, targetVector):
        self._Networks[index].fit(np.asarray(dataVector, dtype="float"), np.asarray(targetVector, dtype="float"))

    # gets a prediction from the network with the given inputc
    def getPrediction(self, inNormalisedData, inIdentifier):
        pred = 0
        for index, i in enumerate(NaturalLanguageObject._Identifiers):
            # get the index
            if(inIdentifier == i):
                # if the vocabulary is empty for this index return 0
                if(len(self._Vocabulary[index]) > 0):
                    pred = self._Networks[index].predict(inNormalisedData)
                else:
                    return 0
        if(pred == 0):
            return 0
        return float(round(pred[0], _MAX_DECIMAL_PLACES))
    # returns the probability of the prediction
    def getPredictionProbability(self, inNormalisedData, inIdentifier):
        pred = [[0]]
        for index, i in enumerate(NaturalLanguageObject._Identifiers):
            # get the index
            if(inIdentifier == i):
                # if the vocabulary is empty for this index return 0
                if(len(self._Vocabulary[index]) > 0):
                    return self._Networks[index].predict_proba(inNormalisedData)
                else:
                    return pred
        return pred
    # Returns the predicted normal in the form of a word
    def getPredictedWord(self, inNormalisedData, inIdentifier):
        pred = self.getPrediction(inNormalisedData, inIdentifier)
        return self._getFromVocab(inIdentifier, pred)

    def __init__(self):
        # Create a sperate network for each identifier
        for index in range(0, len(NaturalLanguageObject._Identifiers)):
            #针对NaturalLanguageObject._Identifiers中的每个分类，分别创建一个KNN分类器。
            nn = KNeighborsClassifier()
            #将创建的KNN分类器存入到_Networks[]中
            self._Networks.append(nn)
        # Create th etraining sets for the multiple svm networks
        # for _ in range() , 其中　_　就是个匿名变量，在for去遍历range()时，每个遍历到的值都会被放入到　_　中去。
        # 而　[list() for _ in range(number)]这样的用法表示生成一个二维数组，里面有number个空的数组
        # [[],[],[],[],...,[]]
        self.trainingData = [list() for _ in range(len(NaturalLanguageObject._Identifiers))]
        self.trainingDataResults = [list() for _ in range(len(NaturalLanguageObject._Identifiers))]
        self._Vocabulary = [list() for _ in range(len(NaturalLanguageObject._Identifiers))]
