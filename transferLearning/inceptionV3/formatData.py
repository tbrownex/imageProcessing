import random
from sklearn.model_selection import train_test_split

''' Create Train and Validation sets:
   1. Names of the files where we get the image data (bottleneck values)
   2. labels for each image
   '''
def formatData(config, imageClass, classIdx):
    imageFiles = list(imageClass.keys())
    random.shuffle(imageFiles)
    labels = [classIdx[imageClass[f]] for f in imageFiles]
    data = list(zip(imageFiles, labels))
    
    trainData, valData = train_test_split(data, test_size=config["valPct"])
    
    d = {}
    d["trainX"] = [x[0] for x in trainData]
    d["trainY"] = [x[1] for x in trainData]
    d["valX"] = [x[0] for x in valData]
    d["valY"] = [x[1] for x in valData]
    return d