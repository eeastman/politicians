import os
import numpy as np
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score

import vgg16
import utils
import load



# SVM for party
# Regression for age and trump score

def run_regression(batch_size, labels):
    X_imgs = np.load('./layers/fc8.npy')
    #X_imgs = np.load('./pool_layers/interactions/pool1.npy')
    X_imgs = X_imgs.reshape((batch_size, -1))
    #y_labs = np.array([1, 1, 1, 0, 0, 0]) # NEED LABELS
    y_labs = np.array(labels)

    lr = LinearRegression()
    lr.fit(X_imgs, y_labs)
    scores = cross_val_score(lr, X_imgs, y_labs, cv=2) # CHANGE WHEN MORE DATA
    # TODO SAVE
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def main():
    directory = './images/'
    img_paths = './id_age.txt'

    batch, batch_size, labels = load.loadImages(directory, img_paths)

    run_regression(batch_size, labels)


main()