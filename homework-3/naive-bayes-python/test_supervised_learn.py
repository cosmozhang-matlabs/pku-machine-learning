import numpy as np
from supervised import NaiveBayesClassifier

dataset = np.genfromtxt("rand_dataset.csv", delimiter=",").T

sample_count = dataset.shape[1]
train_count = int(sample_count * 0.3)

trainset = dataset[:,0:train_count]
testset = dataset[:,train_count:]

classifier = NaiveBayesClassifier(trainset)
classifier.train()

correctCount = 0
totalCount = 0

probs = np.zeros(testset.shape[1])

for i in range(0, testset.shape[1]):
    labelVal = testset[0,i]
    featureVec = testset[1:,i]
    prediction = classifier.predict(featureVec)
    predVal = prediction[0]
    predProb = prediction[1]
    probs[i] = predProb
    print "    ", testset[:,i], ("[Correct]" if (predVal==labelVal) else "[Wrong]  "), predVal, predProb
    if predVal==labelVal: correctCount += 1
    totalCount += 1

print "Correct: %d / %d" % (correctCount, totalCount)
print "log(jointProb): %f" % np.log(probs).sum()
