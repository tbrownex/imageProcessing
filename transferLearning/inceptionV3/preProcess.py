import tensorflow as tf

def preProcess(fileName):
    '''
    - Read the .jpg file
    - Convert from bytes
    - Resize for Inception
    - Normalize
    '''
    rawBytes = tf.io.read_file(fileName)
    img = tf.image.decode_jpeg(rawBytes, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, size=[299,299])
    img /= 255.0
    img = tf.expand_dims(img, 0)
    return img