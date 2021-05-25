
from tensorflow.keras.datasets import cifar100
from keras.models import load_model
from keras.utils import np_utils
# same for all functions
from keras.applications.resnet import preprocess_input as prepRes
from keras.applications.vgg16 import preprocess_input as prepCNN

(xtrain, ytrain), (xtest, ytest) = cifar100.load_data()
ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)


def demoAllRes(xtrain, ytrain, xtest, ytest):
    files = ["preTrainedResNet.h5", "preTrainedResNet50ELU.h5", "preTrainedResNet50AllAugELU.h5",
             "preTrainedResNet50AllAugELU5.h5",
             "preTrainedResNet101ELUV2.h5", "preTrainedResNet152ELU.h5"]
    xtrain = prepRes(xtrain)
    xtest = prepRes(xtest)

    for file in files:
        model = load_model(file)
        print(file)
        score = model.evaluate(xtrain, ytrain)
        print("Train loss: " + str(score[0]))
        print("Train accuracy: " + str(score[1]))
        print()
        score = model.evaluate(xtest, ytest)
        print("Test loss: " + str(score[0]))
        print("Test accuracy: " + str(score[1]))
        print()
def demoAllCNN(xtrain, ytrain, xtest, ytest):
    files = ["scratchCNN.h5", "preTrainedVGG16.h5", "preTrainedVGG19.h5"]
    xtrain = prepCNN(xtrain)
    xtest = prepCNN(xtest)

    for file in files:
        model = load_model(file)
        print(file)
        score = model.evaluate(xtrain, ytrain)
        print("Train loss: " + str(score[0]))
        print("Train accuracy: " + str(score[1]))
        print()
        score = model.evaluate(xtest, ytest)
        print("Test loss: " + str(score[0]))
        print("Test accuracy: " + str(score[1]))
        print()

# reduce to smaller subset
print("Hi! This runs all the models on a subset of the training and testing data, and tells you the classification accuracy and loss for both")
num = 100
xtest = xtest[0:num, :, :, :]
ytest = ytest[0:num]

xtrain = xtrain[0:num, :, :, :]
ytrain = ytrain[0:num]

demoAllCNN(xtrain,ytrain, xtest, ytest)
demoAllRes(xtrain,ytrain, xtest, ytest)