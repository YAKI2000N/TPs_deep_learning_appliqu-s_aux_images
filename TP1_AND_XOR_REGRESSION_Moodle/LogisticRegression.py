import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import accuracy_score
#####################################################################################################################
#####################################################################################################################
# TODO - Application 3 - Step 5 - Create the ANN model
def modelDefinition():

    # TODO - Application 3 - Step 5a - Define the model as a Sequential model
    model = Sequential()
    # TODO - Application 3 - Step 5b - Add a Dense layer with 8 neurons to the model
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', activation='relu'))
    # TODO - Application 3 - Step 5c - Add a Dense layer (output layer) with 1 neuron
    model.add(Dense(1,kernel_initializer='normal')) 
    # TODO - Application 3 - Step 5d - Compile the model by choosing the optimizer(adam) ant the loss function (MSE)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
#####################################################################################################################
#####################################################################################################################
def main():

    # TODO - Application 3 - Step 1 - Read data from "Houses.csv" file
    csvFile = pd.read_csv("./Houses.csv").values 

    # TODO - Application 3 - Step 2 - Shuffle the data
    np.random.shuffle(csvFile)
    # TODO - Application 3 - Step 3 - Separate the data from the labels (x_data / y_data)

###### This part have been used in all the exercises except the last one (exercise 10 where changes have been made)#####
    # x_data = csvFile[:, :-1] #extract all columns of the csv file except the last one 
    #y_data = csvFile[:, -1]#extract only the last column which is the medValue
#######################################################################################################################
#changes made in exercise 10############################################################
    x_data = csvFile[:, np.r_[0:4, 5:14]] #extract all columns of the csv file except the fifth column it means the axis column
    #np.r_ is a NumPy function that concatenates ranges or arrays along the first axis
    y_data = csvFile[:, 4] #extract the last column which is the medValue 
########################################################################################

    # TODO - Application 3 - Step 4 - Separate the data into training/testing dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    # TODO - Application 3 - Step 5 - Call the function "modelDefinition"
    model = modelDefinition()
    # TODO - Application 3 - Step 6 - Train the model for 100 epochs and a batch of 16 samples
    model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2)

    # TODO - Application 3 - Step 7 - Predict the house price for all the samples in the testing dataset
    predictions = model.predict(x_test)
    # TODO - Exercise 8 - Compute the MSE for the test data
    mse = mse = mean_squared_error(y_test, predictions)
    print("Mean Square Error = {}".format(mse))


    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################