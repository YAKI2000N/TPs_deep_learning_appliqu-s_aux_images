# Application 1 - Step 1 - Import the dependencies
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import layers
from keras.layers import Dropout
from matplotlib import pyplot
from keras.models import load_model
import cv2
import time

# Define the function to summarize learning curves and performances
def summarizeLearningCurvesPerformances(histories, accuracyScores):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='green', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='red', label='test')

        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='green', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='red', label='test')

        # Print accuracy for each split
        print("Accuracy for set {} = {}".format(i, accuracyScores[i]))

    pyplot.show()

    print('Accuracy: mean = {:.3f} std = {:.3f}, n = {}'.format(np.mean(accuracyScores) * 100, np.std(accuracyScores) * 100, len(accuracyScores)))

# Define the function to prepare the data
def prepareData(trainX, trainY, testX, testY):
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = trainX.astype('float32') / 255
    testX = testX.astype('float32') / 255
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

#we used the same CNN architecture as the application 1
def defineModel(input_shape, num_classes):
    model = keras.Sequential()
    #we added a parameter padding = same and incrasing the number of filters in the line below as indicated in exercise9 application2
    model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the function to train and evaluate using k-fold cross-validation
def defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY):
    #Application 2 - Step 2 - Prepare the cross validation datasets
    k_folds = 5
    accuracyScores = []
    histories = []
    kfold = KFold(k_folds, shuffle=True, random_state=1)
    for train_idx, val_idx in kfold.split(trainX):
        #TODO - Application 2 - Step 3 - Select data for train and validation
        trainX_i, trainY_i = trainX[train_idx], trainY[train_idx]
        valX_i, valY_i = trainX[val_idx], trainY[val_idx]
        #TODO - Application 2 - Step 4 - Build the model - Call the defineModel function
        model = defineModel((28, 28, 1), 10)
        #TODO - Application 2 - Step 5 - Fit the model
        history = model.fit(trainX_i, trainY_i, epochs=5, batch_size=32, validation_data=(valX_i, valY_i), verbose=1)
        #TODO - Application 2 - Step 6 - Save the training related information in the histories list
        histories.append(history)
        #TODO - Application 2 - Step 7 - Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(testX, testY, verbose=0)
        #TODO - Application 2 - Step 8 - Save the accuracy in the accuracyScores list
        accuracyScores.append(accuracy)
    return histories, accuracyScores


def main():
    #we used the same code as the application 1 for the data preparation
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    print('Train:', trainX.shape, trainY.shape)
    print('Test:', testX.shape, testY.shape)
    trainX, trainY, testX, testY = prepareData(trainX, trainY, testX, testY)

    #TODO - Application 2 - Step 1 - Define, train and evaluate the model using K-Folds strategy
    histories, accuracyScores = defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY)

    #TODO - Application 2 - Step9 - System performance presentation
    summarizeLearningCurvesPerformances(histories, accuracyScores)

if __name__ == '__main__':
    main()
