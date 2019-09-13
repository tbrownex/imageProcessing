import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from getConfig import getConfig
from preProcess import preProcess

def getFilenames(config):
    ''' Create a .npy for any .jpg that are missing its .npy '''
    fileNames = []
    for root, _, files in tf.io.gfile.walk(config["dataLoc"]):
        for f in files:
            fileNames.append(os.path.join(root, f))
    jpgs = [x for x in fileNames if ".jpg" in x]
    npys = [x for x in fileNames if ".npy" in x]
    npys = [x[:-4] for x in npys]
    jpgs = [x[:-4] for x in jpgs]
    fileNames = [x+".jpg"  for x in jpgs if x not in npys]
    return fileNames

def createDataset(fileNames):
    ''' Create a dataset of all the image files (.jpg)
    'map' does the resizing and normalizing '''
    ds = tf.data.Dataset.from_tensor_slices(fileNames)
    return ds.map(preProcess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
def getBottleneck(config, img):
    ''' For each image, get the Feature Vector/bottleneck values and return in a list '''
    # Get the Inception network
    inception = hub.Module(config["modelURL"])
    bottleneck = inception(img)
    bottlenecks = []
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        try:
            while True:
                bottlenecks.append(sess.run(bottleneck))
        except tf.errors.OutOfRangeError:
            pass
    return bottlenecks
    
def writeBottlenecks(fileNames, bottlenecks):
    # Associate the name of the file with its bottleneck layer
    bottlenecks=np.array(bottlenecks)
    z = zip(fileNames, bottlenecks)
    for file, data in z:
        file = file.replace(".jpg", "")
        np.save(file, data)   # Creates a .npy
        
#def createBottlenecks():
if __name__ == "__main__":
    ''' For each image file (.jpg), do some pre-processing to speed up training. Basically that means running the image through the Inception model and getting the Feature Vector aka Bottleneck Layer, which is compute-intensive.
    So get the Feature Vector and write it out to a file...one file for each image. The output file is type ".npy" which is faster than reading a ".csv"
    '''
    config = getConfig()
    fileNames = getFilenames(config)
    ds = createDataset(fileNames)
    iter = ds.make_one_shot_iterator()
    img = iter.get_next()
    bottlenecks = getBottleneck(config, img)
    writeBottlenecks(fileNames, bottlenecks)