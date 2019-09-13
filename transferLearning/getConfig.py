''' A dictionary object holds key parameters'''

__author__ = "Tom Browne"

def getConfig():
    d = {}
    d["dataLoc"]     = "/home/tbrownex/data/flower_photos/"
    d["modelURL"] = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3"
    d["npyFileHeaderSize"] = 32
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "transferLearning.log"
    d["logDefault"] = "info"
    d["batchSize"] = 32
    d["valPct"]     = 0.15
    d["TBdir"] = '/home/tbrownex/TF/Tensorboard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/TF/models/"  # where to save models
    return d