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
#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
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

        #print accuracy for each split
        print("Accuracy for set {} = {}".format(i, accuracyScores[i]))

    pyplot.show()

    print('Accuracy: mean = {:.3f} std = {:.3f}, n = {}'.format(np.mean(accuracyScores) * 100, np.std(accuracyScores) * 100, len(accuracyScores)))
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################

def prepareData(trainX, trainY, testX, testY):
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainX = trainX.astype('float32') / 255
    testX = testX.astype('float32') / 255

    # Convert target labels to one-hot encoded format
    trainY = to_categorical(trainY, num_classes=10)
    testY = to_categorical(testY, num_classes=10)

    return trainX, trainY, testX, testY

#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
#def defineModel(input_shape, num_classes, num_filters):#used for exercise 2
#def defineModel(input_shape, num_classes):used for exercise 3
#def defineModel(input_shape, num_classes, learning_rate):used for exercise 5
#def defineModel(input_shape, num_classes, dropout_percentage):used for exercise 6

def defineModel(input_shape, num_classes):

    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = keras.Sequential()

    #TODO - Application 1 - Step 6b - Create the first hidden layer as a convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))#use the optimal value of filters number

    #model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))#used for exercise2

    #TODO - Application 1 - Step 6c - Define the pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    #model.add(Dropout(dropout_percentage)) : used for exercise 6

    #TODO - Application 1 - Step 6d - Define the flatten layer
    model.add(layers.Flatten())

    #TODO - Application 1 - Step 6e - Define a dense layer of size 16
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform')) #use the optimal value of neurons number
    #model.add(layers.Dense(neurons, activation='relu', kernel_initializer='he_uniform'))

    #TODO - Application 1 - Step 6f - Define the output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    #TODO - Application 1 - Step 6g - Compile the model
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
#####################################################################################################################
##################Loading image for exercise 8#######################################################################

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

#####################################################################################################################
#####################################################################################################################
#def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, neurons):used for exercise 3 
#def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, epoch):used for exercise 4
#def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, learning_rate):used for exercise 5
#def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, droupout_percentage):used for exercise 6

def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY):

    #TODO - Application 1 - Step 6 - Call the defineModel function
    #start of training : used to calculate the training time
    #start_time=time.time()

    #model = defineModel((28, 28, 1), 10, num_filters): used for exercise 2
    #model = defineModel((28, 28, 1), 10) #used for exercise 3
    #model = defineModel((28, 28, 1), 10, droupout_percentage) : used for exercise 6
    model = defineModel((28, 28, 1), 10)

########################## used for exercise 2 and 3  ################################################################################""
    #this code is used to solve the second exercise to calulate the convergence time and the system accuracy with different filters 
    #epochs = 5
    #prev_accuracy = 0
    #start_time = time.time()
    #for epoch in range(epochs):
        #history = model.fit(trainX, trainY, epoch=1, batch_size=32, validation_data=(testX, testY), verbose=1)
        #_, accuracy = model.evaluate(testX, testY, verbose=0)
        #if np.abs(accuracy - prev_accuracy) < 0.001:  # Convergence criteria
         #   break
        #prev_accuracy = accuracy
#####################################################################################################################

    #TODO - Application 1 - Step 7 - Train the model
    model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1) #change to optimal value of epoch to 10

    model.save('./Fashion_MNIST_model.h5')
    #Load the pretrained model 
    model=load_model('./Fashion_MNIST_model.h5')
   
    
    #TODO - Application 1 - Step 8 - Evaluate the model
    loss, accuracy = model.evaluate(testX, testY, verbose=0)


    #used to calculate the training time
    #end_time=time.time()

    #print('Number of Filters: {}'.format(num_filters)) : used for exercise2
    #print('Number of neurons: {}'.format(neurons))  #used for exercise3
    #print('Train the model with :', epoch, 'epoch')#used for exercise4
    #print('Train the model with :', learning_rate, 'as a learning rate')#used for exercise5
    #print('Train the model with :', droupout_percentage, 'as a deopout percentage')#used for exercise 6

    print('Test Accuracy: %.3f' % (accuracy * 100.0))

    #convergence_time = end_time-start_time
    #print("Convergence Time: {:.2f} seconds".format(convergence_time))
#################################################################################""
###this part is used to print the training time in exercise 1### 
    #print("Training Time (CPU): {:.2f} seconds".format(end_time - start_time))


    return model
    
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY):

    k_folds = 5

    accuracyScores = []
    histories = []

    #Application 2 - Step 2 - Prepare the cross validation datasets
    kfold = KFold(k_folds, shuffle=True, random_state=1)

    for train_idx, val_idx in kfold.split(trainX):

        #TODO - Application 2 - Step 3 - Select data for train and validation
        trainX_i, trainY_i = trainX[train_idx], trainY[train_idx]
        valX_i, valY_i = trainX[val_idx], trainY[val_idx]
        #Firstly this has generated error so we added the lines below 
        trainY_i = to_categorical(trainY_i)  # Convert to one-hot encoded categorical labels
        valY_i = to_categorical(valY_i)      # Convert to one-hot encoded categorical labels

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
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 2 - Load the Fashion MNIST dataset in Keras
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data() 

    #TODO - Application 1 - Step 3 - Print the size of the train/test dataset
    print('Train:', trainX.shape, trainY.shape)
    print('Test:', testX.shape, testY.shape)
    

    #TODO - Application 1 - Step 4 - Call the prepareData method
    model = prepareData(trainX, trainY, testX, testY)

    #TODO - Application 1 - Step 5 - Define, train and evaluate the model in the classical way
    model=defineTrainAndEvaluateClassic(trainX, trainY, testX, testY)

    #load the query image using the method above 
    query_img = load_image('sample_image.png')

    #generate predictions for the query image and return probabilities for each class 
    prediction = model.predict(query_img)

    #used to find the index of the highest value in the prediction array
    predicted_class = np.argmax(prediction)

    #define the classes labels 

    classes_labels= {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

    #assigns the predicted class to its category

    category= classes_labels[predicted_class]

    print('Predicted Category for the query image is :', category)
   

    ########################a loop used to solve the exercise 2 in order to change the number of filter##########################"
    #num_filters_list = [8, 16, 32, 64, 128]
    #for num_filters in num_filters_list:
        #defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, num_filters)
    ###########################################################################################################################

    ########################a loop used to solve the exercise 3 in order to change the number of neurons##########################
    #neurons = [16, 64, 128, 256, 512]
    #for neuron in neurons:
     #   defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, neuron)
    ###########################################################################################################################

    ########################a loop used to solve the exercise 4 in order to change the number of epochs##########################
    #epochs = [1, 2, 5, 10, 20]
    #for epoch in epochs:
     #   defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, epoch)
    ###########################################################################################################################

    ########################a loop used to solve the exercise 5 in order to change the learning rate of SGD optimize##########################
    #learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    #for learning_rate in learning_rates:
     #   defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, learning_rate)
    ###########################################################################################################################

     ########################a loop used to solve the exercise 6 in order to change the dropout percentage#########################
    #dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
    #for dropout in dropouts:
     #   defineTrainAndEvaluateClassic(trainX, trainY, testX, testY, dropout)
    ###########################################################################################################################


#######################################################################################################
#######################This part respnds to exercise 8 ###############################################





################################################################################################################""




#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
