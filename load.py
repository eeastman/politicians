import os
import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score

import vgg16
import utils

def getPoolingLayers(batch, batch_size):
    layerPath = './layers/' # CHANGE SO YOU DON'T OVERRIDE

    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [batch_size, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            fc8 = sess.run(vgg.fc8, feed_dict=feed_dict)
            np.save(os.path.join(layerPath + 'fc8.npy'),fc8)

            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            print(prob)

            return prob

def loadImages(directory, img_paths):
    if img_paths != '':
        f = open(img_paths,'r')
        img_paths = f.read().split('\n')
    else:
        img_paths = sorted(os.listdir(directory))
    images = []
    labels = []
    batch_size = 0

    for path in img_paths:
        if path == '':
            continue
        batch_size += 1
        img_path = directory + path[:-2]
        img = utils.load_image(img_path)
        resize = img.reshape((1, 224, 224, 3))
        images.append(resize)
        labels.append(path[-2:])
    print('labels', labels)

    batch = np.zeros((batch_size, 224, 224, 3))
    for i, image in enumerate(images):
        batch[i, :, :, :] = image

    return batch, batch_size, labels

def main():
    directory = './images/'
    img_paths = './id_party.txt'

    batch, batch_size, labels = loadImages(directory, img_paths)
    x = getPoolingLayers(batch, batch_size)

    # model = VGG16(weights='imagenet', include_top=False)

    # x = batch[1, :, :]
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # features = model.predict(x)
    # print('Predicted:', decode_predictions(features, top=3)[0])
main()