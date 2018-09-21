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

# SVM for party
# Regression for age and trump score

def run_svm(batch_size, labels):
    X_imgs = np.load('./pca_layers/interactions/pool1.npy')
    #X_imgs = np.load('./pool_layers/interactions/pool1.npy')
    X_imgs = X_imgs.reshape((batch_size, -1))
    #y_labs = np.array([1, 1, 1, 0, 0, 0]) # NEED LABELS
    y_labs = np.array(labels)
    clf = svm.SVC(kernel='linear', C=1).fit(X_imgs, y_labs)
    scores = cross_val_score(clf, X_imgs, y_labs, cv=2) # CHANGE WHEN MORE DATA
    # TODO SAVE
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def main():
    directory = './images/'
    img_paths = './id_party.txt'

    batch, batch_size, labels = loadImages(directory, img_paths)


main()