import tensorflow
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

    # active = 'relu'
    active = 'elu'
    # make basic CNN
    model = tensorflow.keras.Sequential()
    # input shape is fixed, just hard code
    input_shape = (32, 32, 3)
    # 2 layers of conv with max pooling
    model.add(Conv2D(32, kernel_size=(3, 3), activation=active, padding="same", input_shape=input_shape))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='elu', input_shape=input_shape))
    # model.add(Dropout(0.2))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation=active))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation=active, padding="same"))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='elu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation=active))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation=active, padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation=active, padding="same"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(1152, kernel_size=(3, 3), activation=active, padding="same"))

    model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))
    model.add(Conv2D(256, kernel_size=(3, 3), activation=active, padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))
    model.add(Conv2D(256, kernel_size=(3, 3), activation=active, padding="same"))
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    # flatten to linear data
    model.add(Flatten())
    # send through dense layer
    model.add(Dense(400, activation='relu'))
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

    dataAug = ImageDataGenerator(
        featurewise_center=True,
        zca_whitening=True,
        horizontal_flip=True,
    )

    dataAug.fit(xtrain)

    # don't do horizontal flip
    testAug = ImageDataGenerator(
        featurewise_center=True,
        zca_whitening=True,
    )
    testAug.fit(xtest)

    model.fit(dataAug.flow(xtrain, ytrain,
        batch_size=32),
        epochs=25,
        verbose=True,
        validation_data=testAug.flow(xtest, ytest)
        # callbacks=[early_stopping_monitor],
    )

    score = model.evaluate(testAug.flow(xtest, ytest))
    # print("Test accuracy:" + str(score))
    print("Test loss: " + str(score[0]))
    print("Test accuracy: " + str(score[1]))


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = loadAndPrep100()

    CNN(xtrain, ytrain, xtest, ytest)
