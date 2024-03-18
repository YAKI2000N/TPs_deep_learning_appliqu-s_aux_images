import numpy as np

#def sigmoid(n):

    # Define the sigmoid function as activation function
    #return 1.0/(1.0 + np.exp(-n))

#def sigmoidDerivative(n):
    #return n*(1-n)

def tanh(n):  #changing sigmoid to tanh function 

    # Define the tanh function as activation function
    return np.tanh(n)
#####################################################################################################################
#####################################################################################################################
def tanhDerivative(n):
    # The derivative of tanh
    return 1 - np.tanh(n) ** 2
#####################################################################################################################
#####################################################################################################################

def forwardPropagationLayer(p, weights, biases):

    a = None  # the layer output

    # Multiply weights with the input vector (p) and add the bias   =>  n
    n = np.dot(p, weights) + biases

    # Pass the result to the activation function  =>  a
    #a=sigmoid(n)
    a = tanh(n)  #remplacer la sigmoid avec tanh 

    return a
#####################################################################################################################
#####################################################################################################################
# This function compute the prediction error between the true labels and the predictions
def computePredictionError(labels, predictions):
    N = len(labels)
    error = np.sum((labels - predictions) ** 2) / (2 * N)
    return error


#####################################################################################################################
#####################################################################################################################
def main():

    #Application 2 - Train a ANN in order to predict the output of an XOR gate.
    #The network should receive as input two values (0 or 1) and should predict the target output

    #Input data
    points = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    #Labels
    labels = np.array([[0], [1], [1], [0]])

    # Initialize the weights and biases with random values
    inputSize = 2
    noNeuronsLayer1 = 2
    noNeuronsLayer2 = 1

    weightsLayer1 = np.random.uniform(size=(inputSize, noNeuronsLayer1))
    weightsLayer2 = np.random.uniform(size=(noNeuronsLayer1, noNeuronsLayer2))

    biasLayer1 = np.random.uniform(size=(1, noNeuronsLayer1))
    biasLayer2 = np.random.uniform(size=(1, noNeuronsLayer2))


    noEpochs = 5000
    learningRate = 0.3
    min_error = 0.01


    # Train the network for noEpochs
    while True:
        # Forward Propagation
        hidden_layer_output = forwardPropagationLayer(points, weightsLayer1, biasLayer1)
        predicted_output = forwardPropagationLayer(hidden_layer_output, weightsLayer2, biasLayer2)
        # Compute Prediction Error with the fucntion defined above
        error = computePredictionError(labels, predicted_output)
        #A condition that halts the training process when the error falls below the minimum allowable error.
        if error < min_error:
            break
        # Backpropagation
        bkProp_error = labels - predicted_output
        # d_predicted_output = bkProp_error * sigmoidDerivative(predicted_output)
        d_predicted_output = bkProp_error * tanhDerivative(predicted_output) 
        error_hidden_layer = d_predicted_output.dot(weightsLayer2.T)
        # d_hidden_layer = error_hidden_layer * sigmoidDerivative(hidden_layer_output)
        d_hidden_layer = error_hidden_layer * tanhDerivative(hidden_layer_output)
        # Updating Weights and Biases
        weightsLayer2 = weightsLayer2 + hidden_layer_output.T.dot(d_predicted_output) * learningRate
        biasLayer2 = biasLayer2 + np.sum(d_predicted_output, axis=0, keepdims=True) * learningRate
        weightsLayer1 = weightsLayer1 + points.T.dot(d_hidden_layer) * learningRate
        biasLayer1 = biasLayer1 + np.sum(d_hidden_layer, axis=0, keepdims=True) * learningRate
        #this instruction counts the number of epochs 
        noEpochs += 1
    #Displaying the result after achieving a low prediction error, as defined in the condition above.
    print("Minimum number of epochs to achieve prediction error < 0.01:", noEpochs)
    # Print weights and bias
    print("weightsLayer1 = {}".format(weightsLayer1))
    print("biasesLayer1 = {}".format(biasLayer1))

    print("weightsLayer2 = {}".format(weightsLayer2))
    print("biasLayer2 = {}".format(biasLayer2))

    # Display the results
    for i in range(len(labels)):
        outL1 = forwardPropagationLayer(points[i], weightsLayer1, biasLayer1)
        outL2 = forwardPropagationLayer(outL1, weightsLayer2, biasLayer2)

        print("Input = {} - Predict = {} - Label = {}".format(points[i], outL2, labels[i]))
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == "__main__":
    main()
#####################################################################################################################
#####################################################################################################################

