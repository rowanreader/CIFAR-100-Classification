import tensorflow
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math
import numpy as np
import matplotlib.pyplot as plt
# from https://github.com/mwv/zca
from zca import ZCA



# make custom activation function
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from keras.utils.generic_utils import get_custom_objects

# loads and prepares data from cifar 100 dataset
# prep = reshapes, normalizes
# returns all train and test data
def loadAndPrep100():
    # x data in form of (num samples, num channels, width, height) =  (50000 or 10000, 32, 32, 3)
    # both in uint8
    (xtrain, ytrain), (xtest, ytest) = cifar100.load_data()

    # use one-hot encoding for y
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)

    # covert x data from unsigned into to float, then normalize
    xtrain = xtrain.astype('float32')/255
    xtest = xtest.astype('float32')/255


    return xtrain, ytrain, xtest, ytest

def CNN(xtrain, ytrain, xtest, ytest):
    # make basic CNN
    model = tensorflow.keras.Sequential()
    # input shape is fixed, just hard code
    input_shape = (32, 32, 3)
    # 2 layers of conv with max pooling
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", input_shape=input_shape))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='elu', input_shape=input_shape))
    # model.add(Dropout(0.2))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='elu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten to linear data
    model.add(Flatten())
    # send through dense layer
    model.add(Dense(1024, activation='relu'))
    # output layer, num classes is 100, use softmax for output
    model.add(Dense(100, activation='softmax'))

    # loss and optimizer decided based on literature
    model.compile(loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    model.fit(xtrain, ytrain,
        batch_size=32,
        epochs=100,
        verbose=True,
        # callbacks=[early_stopping_monitor],
    )

    score = model.evaluate(xtest, ytest, verbose=0)
    print("Test loss: " + str(score[0]))
    print("Test accuracy: " + str(score[1]))

# applies global contrast normalization, preprocessing step
# removes mean, multiplies by constant and divide by small number
# small number is the sqrt of the new mean squared plus a constant, or a constand alone, which ever is larger
# takes in array of images, returns array w all images preprocessed with this
# from:
# https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python
def globalContrastNormalization(images, s=1, lmda=10, epsilon=1e-5):

    num = len(images)
    for i in range(num):
        img = images[i]

        ave = img.mean(axis=0)
        img = img - ave
        img = img / np.sqrt((img ** 2).sum(axis=1))[:, None]

        # contrast = np.sqrt(lmda + np.mean(img**2))
        # img = s * img/max(contrast, epsilon)
        images[i] = img

        # m, M = img.min(), img.max()
        # plt.imshow((img - m) / (M - m))
        # plt.imshow(img)
        # plt.show()
    return images

# applies zca whitening on image as preprocessing
# takes in array of images, returns array w all images preprocessed with this
# code from https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
def zca(img):
    img = img.reshape(-1, 3072)
    # get covariance matrix of images
    covariance = np.cov(img, rowvar=True)
    # apply svd to get U (eigvecs of cov), S (eigvals of cov) and V (transpose of U)
    U, S, V = np.linalg.svd(covariance)
    epsilon = 1e-5 # avoid division by 0
    zca = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    #now apply to each image in list
    num = len(img)
    for i in range(num):
        img[i] = np.dot(zca, img[i])
    plt.imshow(img[6].reshape(32, 32, 3))
    plt.show()
    return img


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = loadAndPrep100()


    # xtrain = globalContrastNormalization(xtrain)
    # xtest = globalContrastNormalization(xtest)
    # xtrain = zca(xtrain)
    # xtest = zca(xtest)

    # xtrain = xtrain.reshape(-1, 3072)
    # xtest = xtest.reshape(-1, 3072)
    # trf = ZCA().fit(xtrain)
    # xtrain = trf.transform(xtrain)
    # trf = ZCA().fit(xtest)
    # xtest = trf.transform(xtest)
    # temp = xtrain[6].reshape((32,32,3))
    # m, M = temp.min(), temp.max()
    # plt.imshow((temp - m) / (M - m))
    # plt.show()

    CNN(xtrain, ytrain, xtest, ytest)
