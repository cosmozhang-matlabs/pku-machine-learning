import numpy as np
from supervised import NaiveBayesClassifier

# Naive Bayes cluster using EM algorithm
# if labeledData is set, it is a half-supervised cluster
# if labelCount is not set, it will guess the best number of labels
class EMNaiveBayesCluster:

    def __init__(self, dataset, labeledData=None, labelCount=None, threshold=0.001):
        if type(dataset) == str: dataset = np.genfromtxt(dataset, delimiter=",").T
        self.dataset = dataset
        dataShape = dataset.shape
        self.sampleCount = dataShape[1]
        self.featureDims = dataShape[0]
        self.labeledData = labeledData
        self.labelCount = None # if not specified, this will be set when training
        self.threshold = threshold
        self.trained = False

        # judge if labeledData valid
        if (not labeledData is None) and (labeledData.shape[0] != (self.featureDims+1)):
            self.labeledData = None
            raise Exception("Features of the labeled dataset should have the same dimension with dataset")

        # variables to be trained
        self.labelValues = None
        self.featureValues = None
        self.labels = None # clustered labels

    def cluster(self):
        self.featureValues = []
        featureValuesCounts = np.zeros((self.featureDims,))
        for fIdx in range(0, self.featureDims):
            self.featureValues.append(np.unique(self.dataset[fIdx,:]))
            featureValuesCounts[fIdx] = self.featureValues[fIdx].shape[0]
        if self.labelCount is None:
            self.labelCount = int(np.round(featureValuesCounts.mean()))
        self.labelValues = np.array(range(1,self.labelCount+1))

        features = self.dataset

        # initial parameters (labels)
        # if labeledData is set, initial labels with a classifier trained with labeldData
        labels = None
        if self.labeledData is None:
            print "using unsupervised"
            labels = np.ceil(np.random.uniform(0, self.labelCount, self.sampleCount))
        else:
            print "using half-supervised"
            labels = np.zeros(self.sampleCount)
            classifier = NaiveBayesClassifier(self.labeledData, trainNow=True)
            for i in range(0,self.sampleCount):
                prediction = classifier.predict(self.dataset[:,i])
                labels[i] = prediction[0]

        # begin iteration
        data = np.zeros((1+self.featureDims, self.sampleCount))
        data[1:,:] = features
        lastProbs = np.zeros(self.sampleCount)
        lastJointProb = 0
        iterCounter = 0
        while True:
            data[0,:] = labels
            classifier = NaiveBayesClassifier(data, trainNow=True)
            probs = np.zeros(self.sampleCount)
            logJointProb = 0
            for i in range(0,self.sampleCount):
                prediction = classifier.predict(data[1:,i])
                predLabel = prediction[0]
                predProb = prediction[1]
                labels[i] = predLabel
                probs[i] = predProb
                logJointProb += np.log(predProb)
                # print "    ", data[:,i], predLabel, predProb
            iterCounter += 1
            print "Iteration: % 3d     log(jointProb): %f" % (iterCounter, logJointProb)
            if np.abs(logJointProb - lastJointProb) < self.threshold: break
            lastJointProb = logJointProb
            lastProbs = probs

        self.trained = True
