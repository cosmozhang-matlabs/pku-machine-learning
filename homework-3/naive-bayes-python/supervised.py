import numpy as np

class NaiveBayesClassifier:

    lambdaConst = 1.0 # Bayes estimation: Laplas smooth

    def __init__(self, dataset, trainNow = False):
        if type(dataset) == str: dataset = np.genfromtxt(dataset, delimiter=",").T
        self.dataset = dataset
        self.trained = False
        dataShape = dataset.shape
        self.sampleCount = dataShape[1]
        self.featureDims = dataShape[0] - 1

        # variables to be trained
        self.labelValues = None
        self.featureValues = None
        self.countXY = None # x with y for each dim of features
        self.countY = None

        if trainNow: self.train()

    def train(self):
        self.labelValues = np.unique(self.dataset[0,:])
        self.featureValues = []
        for fIdx in range(0, self.featureDims):
            self.featureValues.append(np.unique(self.dataset[fIdx+1,:]))

        labelValuesCnt = self.labelValues.shape[0]
        labelIndices = []
        self.countY = np.zeros( (labelValuesCnt,) )
        self.countXY = []
        # Calculate sum of I(Y)
        for iLabel in range(0, labelValuesCnt):
            labelIndices.append( (self.dataset[0,:] == self.labelValues[iLabel]) )
            self.countY[iLabel] = float(np.where(labelIndices[iLabel])[0].shape[0])
        # Calculate sum of I(X,Y)
        for fIdx in range(0, self.featureDims):
            featureValuesCnt = self.featureValues[fIdx].shape[0]
            countXYMat = np.zeros( ( featureValuesCnt, labelValuesCnt ) )
            for iFeature in range(0, featureValuesCnt):
                featureIndice = (self.dataset[fIdx+1,:] == self.featureValues[fIdx][iFeature])
                for iLabel in range(0, labelValuesCnt):
                    jointIndice = np.logical_and(labelIndices[iLabel], featureIndice)
                    countXYMat[iFeature,iLabel] = float(np.where(jointIndice)[0].shape[0])
            self.countXY.append(countXYMat)

        self.trained = True

    # return: tuple( classLabel, probability )
    def predict(self, featureVector):
        lambdaConst = NaiveBayesClassifier.lambdaConst
        if featureVector.shape[0] != self.featureDims:
            raise Exception("the feature vector should have %d dimensions" % self.featureDims)
        labelValuesCnt = self.labelValues.shape[0]
        totalProbs = np.zeros( (labelValuesCnt,) ) + 1
        for fIdx in range(0, self.featureDims):
            featureValue = featureVector[fIdx]
            featureValueIndex = np.where(self.featureValues[fIdx] == featureValue)[0][0]
            probs = ((self.countXY[fIdx][featureValueIndex,:] + lambdaConst) / (self.countY + lambdaConst * self.featureValues[fIdx].shape[0]))
            totalProbs = totalProbs * probs
        maxIndex = totalProbs.argmax()
        maxScore = totalProbs[maxIndex]
        maxLabel = self.labelValues[maxIndex]
        return (maxLabel, maxScore)


