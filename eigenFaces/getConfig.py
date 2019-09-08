''' A dictionary object holds key parameters'''

__author__ = "Tom Browne"

def getConfig():
    d = {}
    d["dataLoc"]     = "/home/tbrownex/data/faces/"
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "eigenFaces.log"
    d["logDefault"] = "info"
    d["batchSize"] = 32
    d["valPct"]     = 0.15
    d["TBdir"] = '/home/tbrownex/TF/Tensorboard'         # where to store Tensorboard data
    d["modelDir"] = "/home/tbrownex/TF/models/"  # where to save models
    return d