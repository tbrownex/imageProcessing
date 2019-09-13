import random
import subprocess
import tensorflow as tf

from getConfig import getConfig

count = 100
path = "gs://ml-datasets1/flower_photos/"
types = ["daisy", "roses", "tulips", "dandelion", "sunflowers"]

def getFileNames(typ):
    part1 = "gsutil ls "
    part2 = path + typ
    files = subprocess.check_output(part1+part2, shell=True)
    files = files.decode("utf-8").split("\n")
    return files

def copyFile(file, target):
    part1 = "gsutil cp "
    part2 = file + " " + target
    subprocess.call(part1+part2, shell=True)

def checkDir(config):
    ''' Make sure the target dir exists; if not, create it '''
    if tf.io.gfile.exists(config["dataLoc"]):
        pass
    else:
        tf.io.gfile.MkDir(config["dataLoc"])

if __name__ == "__main__":
    config = getConfig()
    checkDir(config)
    for typ in types:
        fileNames = getFileNames(typ)
        fileNames = random.sample(fileNames, count)
        target = config["dataLoc"]+typ+"/"
        for file in fileNames:
            copyFile(file, target)
            