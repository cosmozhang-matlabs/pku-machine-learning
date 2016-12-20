import numpy as np

def genRandomData ( sample_count = 10000, save_path = None ):

    label_count = 4
    feature_dims = 5
    feature_mutrate = np.zeros((feature_dims,)) + 0.1

    feature_centers = np.zeros((feature_dims,label_count))
    for i in range(0,feature_dims):
        feature_centers[i,:] = np.array(range(1,label_count+1))

    labels = np.zeros((sample_count,))
    features = np.zeros((feature_dims,sample_count))

    for i in range(0,sample_count):
        new_label = np.ceil(np.random.uniform(0, label_count)) + 2
        new_feature = np.zeros((feature_dims,))
        new_feature[:] = feature_centers[:, label_count-int(new_label)]
        for j in range(0,feature_dims):
            if np.random.uniform(0,1) < feature_mutrate[j]:
                new_feature[j] = np.ceil(np.random.uniform(0, label_count))
        labels[i] = new_label
        features[:,i] = new_feature

    result_data = np.zeros((feature_dims+1,sample_count))
    result_data[0,:] = labels
    result_data[1:,:] = features

    if save_path:
        np.savetxt(save_path, result_data.T, delimiter=",")

    return result_data

if __name__ == "__main__":
    data = genRandomData(save_path = "rand_dataset.csv")
    print "Generated %d samples" % data.shape[1]
