import tensorflow as tf

def loadFeatureVector(fileName, labels):
    ''' Each record returned from the dataset will be run through this function 
    hard-coding two values: "depth" and numpy header size. That's because the signature to this function is ruled by the return values of the dataset iterator
    '''
    data = tf.io.read_file(fileName)
    data = tf.decode_raw(data, out_type=float)
    # depth should be the number of classes
    labels = tf.one_hot(labels, depth=5)
    # hard-coding the numpy file header size: skip the header
    return data[32:], labels

def createTrainDS(dataDict, config):
    ds = tf.data.Dataset.from_tensor_slices((dataDict["trainX"], dataDict["trainY"]))
    ds = ds.map(loadFeatureVector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    return ds.batch(config["batchSize"])

def createValDS(dataDict, config):
    ''' You need the Validation set in batches due to sharing the iterator: iterator requires fixed shape (batchsize in 1st dimension)'''
    ds = tf.data.Dataset.from_tensor_slices((dataDict["valX"], dataDict["valY"]))
    ds =  ds.map(loadFeatureVector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds.batch(config["batchSize"])

def createDataset(dataDict, config, typ):
    assert typ in ["train", "val"], "invalid dataset typ (train or val)"
    if typ == "train":
        return createTrainDS(dataDict, config)
    else:
        return createValDS(dataDict, config)