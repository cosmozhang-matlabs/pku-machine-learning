import numpy as np
from unsupervised import EMNaiveBayesCluster

dataset = np.genfromtxt("rand_dataset.csv", delimiter=",").T
sampleCount = dataset.shape[1]
labeledCount = int(sampleCount*0.003)
labeledDataset = dataset[:,0:labeledCount]
unlabeledDataset = dataset[1:,labeledCount:]

# cluster = EMNaiveBayesCluster(dataset[1:,:], labelCount=4)
# cluster = EMNaiveBayesCluster(dataset[1:,:], labeledData=labeledDataset, labelCount=4)
cluster = EMNaiveBayesCluster(unlabeledDataset, labeledData=labeledDataset, labelCount=4)

cluster.cluster()
