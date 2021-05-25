import tensorflow
from tensorflow.keras.datasets import cifar100
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Input, Lambda, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.applications.resnet import preprocess_input, ResNet50

from keras.layers.normalization import BatchNormalization
import random
from tensorflow.keras.applications.vgg16 import VGG16

# loads and prepares data from cifar 100 dataset
# prep = reshapes, normalizes
# returns all train and test data
def loadAndPrep100():
    # x data in form of (num samples, num channels, width, height) =  (50000 or 10000, 32, 32, 3)
    # both in uint8
    (xtrain, ytrain), (xtest, ytest) = cifar100.load_data()

    # # use one-hot encoding for y
    ytrain = np_utils.to_categorical(ytrain)
    ytest = np_utils.to_categorical(ytest)

    # this is for the resnet only (since vgg was abandoned)
    # changes pixel values to be within a certain range
    xtrain = preprocess_input(xtrain)
    xtest = preprocess_input(xtest)

    return xtrain, ytrain, xtest, ytest

# CNN made from scratch
# note: you may want to change to not-preprocessed. shouldn't hurt to classify as is tho.
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


# takes image, size of side of square to blackout and constant to set it to
def cutout(img, s=5, c=0):
    height, width, _ = img.shape
    # select random location within image
    # can be anywhere, even where cutout goes out of bounds (shown to be important)
    x = random.randint(0,width)
    y = random.randint(0, height)
    # set parts on the image centered on x,y to c
    # find 'start' = top and left
    top = y - s//2
    if top < 0:
        top = 0
    bottom = y + s//2
    if bottom > height:
        bottom = height-1
    left = x - s//2
    if left < 0:
        left = 0

    right = x + s//2
    if right > width:
        right = width-1

    img[top:bottom, left:right, :] = c
    return img


# use pretrained resnet 50
def resNet(xtrain, ytrain, xtest, ytest):
    batch = 32
    datagen = ImageDataGenerator(preprocessing_function=cutout, horizontal_flip=True,
                             width_shift_range=0.3, height_shift_range=0.3, samplewise_center=True)
    datagen.fit(xtrain)

    # get the base pre-trained model, trained on imagenet
    # don't include dense layers - want to modify
    # this is the image size we need (imagenet = 224x224, double 32 3 times ->256)
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    for layer in resnet.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    # must upsample images to get them to appropriate size
    model.add(UpSampling2D())
    model.add(UpSampling2D())
    model.add(UpSampling2D())
    model.add(resnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='elu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy']
                  )
    # for testing purposes
    # num = 100
    # xtest = xtest[0:num, :, :, :]
    # ytest = ytest[0:num]
    #
    # xtrain = xtrain[0:num, :, :, :]
    # ytrain = ytrain[0:num]
    history = model.fit(datagen.flow(xtrain, ytrain, batch_size=batch),
                        epochs=4,
                        verbose=1,
                        validation_data=(xtest, ytest)
    )

    score = model.evaluate(xtest, ytest, verbose=0)
    print(score[1])
    model.save('preTrainedResNet50AllAugELU5.h5')

    # plot and save graphs of accuracy and loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.plot()
    plt.savefig("Accuracy.png")

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.plot()
    plt.savefig("Loss.png")

# use pretrained CNN (VGG-16)
# note: code currently has resnet preprocessing function, you will have to change that to use this
# need to use vgg16 preprocess_input
def pretrainedCNN(xtrain, ytrain, xtest, ytest):
    batch = 32

    datagen = ImageDataGenerator(preprocessing_function=cutout)
    datagen.fit(xtrain)

    cnn = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # freeze layers

    for layer in cnn.layers:
        layer.trainable = False

    model = Sequential()
    # must upsample images to get them to appropriate size
    model.add(UpSampling2D())
    model.add(UpSampling2D())
    model.add(UpSampling2D())
    model.add(cnn)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy']
                  )

    model.fit(datagen.flow(xtrain, ytrain),
                        batch_size=batch,
                        epochs=5,
                        verbose=1,
                        validation_data=(xtest, ytest)
                        )

    score = model.evaluate(xtest, ytest, verbose=0)
    print(score[1])
    model.save('preTrainedCNNcutout.h5')

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = loadAndPrep100()
    resNet(xtrain, ytrain, xtest, ytest)
    # CNN(xtrain, ytrain, xtest, ytest)
    #pretrainedCNN(xtrain, ytrain, xtest, ytest)
