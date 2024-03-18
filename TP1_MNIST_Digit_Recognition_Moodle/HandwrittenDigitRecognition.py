import keras
import tensorflow as tf
import time
from keras.datasets import mnist
#from keras.utils import np_utils
from keras.utils import to_categorical
from keras import layers
#########################################################################
def baseline_model(num_pixels, num_classes, neurons):


    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = tf.keras.models.Sequential()   # Modify this


    #TODO - Application 1 - Step 6b - Define a hidden dense layer with 8 neurons
    model.add(layers.Dense(neurons, input_dim=num_pixels, kernel_initializer='normal', activation='relu')) # the number of neurons has to be modified here
    # for exercise 1


    #TODO - Application 1 - Step 6c - Define the output dense layer
    model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))


    # TODO - Application 1 - Step 6d - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # the following line has been used to solve exercise 3:
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse'])


    return model
#########################################################################


def trainAndPredictMLP(train_images, train_labels, test_images, test_labels, neurons, batch_size):


    #TODO - Application 1 - Step 3 - Reshape the MNIST dataset - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = train_images.shape[1] * train_images.shape[2]
    train_images = train_images.reshape((train_images.shape[0], num_pixels)).astype('float32')
    test_images = test_images.reshape((test_images.shape[0], num_pixels)).astype('float32')


    #TODO - Application 1 - Step 4 - Normalize the input values
    train_images = train_images / 255
    test_images = test_images / 255


    #TODO - Application 1 - Step 5 - Transform the classes labels into a binary matrix
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    num_classes = test_labels.shape[1]


    #TODO - Application 1 - Step 6 - Build the model architecture - Call the baseline_model function
 
    model = baseline_model(num_pixels, num_classes, neurons)  #Modify this
   


    #TODO - Application 1 - Step 7 - Train the model
    start_time=time.time()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=batch_size, verbose=2)
    end_time=time.time()


    # the following line has been used to solve exercise 4
    model.save_weights('./application1.h5')


    convergence_time=end_time-start_time
    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(test_images, test_labels, verbose=2)
    print("Baseline Error: {:.2f}".format(100-scores[1]*100))
    print("the convergence time is", convergence_time, "seconds")


    return
#########################################################################


def CNN_model(input_shape, num_classes, size, neurons):


    # TODO - Application 2 - Step 5a - Initialize the sequential model
    model = tf.keras.models.Sequential()   #Modify this


    #TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer
    # model.add(layers.Conv2D(8, size, activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(30, size, activation='relu', input_shape=(28, 28, 1)))
    #the line above has been modified for exercise 9


    #TODO - Application 2 - Step 5c - Define the pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2,2)))


    #the two following lines have been added in Exercise 9
    model.add(layers.Conv2D(15, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))


    #TODO - Application 2 - Step 5d - Define the Dropout layer
    model.add(layers.Dropout(rate=0.2))


    #TODO - Application 2 - Step 5e - Define the flatten layer
    model.add(layers.Flatten())


    #TODO - Application 2 - Step 5f - Define a dense layer of size 128
    # model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(neurons, activation='relu'))


    #the following line has been added in Exercise 9
    model.add(layers.Dense(50, activation='relu'))


    #TODO - Application 2 - Step 5g - Define the output layer
    model.add(layers.Dense(num_classes, activation='softmax'))


    #TODO - Application 2 - Step 5h - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model
#########################################################################


def trainAndPredictCNN(train_images, train_labels, test_images, test_labels, size, neurons, epochs):


    #TODO - Application 2 - Step 2 - reshape the data to be of size [samples][width][height][channels]
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32')
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32')


    #TODO - Application 2 - Step 3 - normalize the input values from 0-255 to 0-1
    train_images = train_images / 255
    test_images = test_images / 255


    #TODO - Application 2 - Step 4 - One hot encoding - Transform the classes labels into a binary matrix
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)


    #TODO - Application 2 - Step 5 - Call the cnn_model function
    model = CNN_model(input_shape=(28,28,1), num_classes=10, size=size, neurons=neurons)  #Modify this




    #TODO - Application 2 - Step 6 - Train the model
    start_time=time.time()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=epochs, batch_size=200, verbose=1)
    end_time=time.time()


    convergence_time=end_time-start_time
    #TODO - Application 2 - Step 8 - Final evaluation of the model - compute and display the prediction error
    scores = model.evaluate(test_images, test_labels, verbose=2)
    print("Baseline Error: {:.2f}".format(100-scores[1]*100))
    print("the convergence time is", convergence_time, "seconds")


    return
#########################################################################


def main():


    #TODO - Application 1 - Step 1 - Load the MNIST dataset in Keras
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


    #TODO - Application 1 - Step 2 - Train and predict on a MLP - Call the trainAndPredictMLP function
    # loop used to solve exercise 1
    # neurons =[8,16,32,64,128]
    # for neuron in neurons :
    #    print("system performance with :", neuron,"neurons")
    #    mlp = trainAndPredictMLP(train_images, train_labels, test_images, test_labels, neuron, batch_size=200)