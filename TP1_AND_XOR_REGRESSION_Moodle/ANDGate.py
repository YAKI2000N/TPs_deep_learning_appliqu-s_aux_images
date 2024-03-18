import math 
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def activationFunction(n):

    # Define the sigmoid activation function
    return 1 / (1 + math.exp(-n))

#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def forwardPropagation(p, weights, bias):

    a = None # the neuron output

    # TODO - Application 1 - Step 4a - Multiply weights with the input vector (p) and add the bias   =>  n
    n = weights[0] * p[0] + weights[1] * p[1] + bias


    # TODO - Application 1 - Step 4c - Pass the result to the activation function  =>  a
    a = activationFunction(n)


    return a
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.
    #The network should receive as input two values (0 or 1) and should predict the target output


    #Input data
    P = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

    #Labels
    t = [0, 0, 0, 1]

    #TODO - Application 1 - Step 2 - Initialize the weights with zero  (weights)
    weights = [0, 0]

    #TODO - Application 1 - Step 2 - Initialize the bias with zero  (bias)
    bias = 0

    #TODO - Application 1 - Step 3 - Set the number of training steps  (epochs)
    epochs = 100

    #TODO - Application 1 - Step 4 - Perform the neuron training for multiple epochs
    for ep in range(epochs):

        valid_predictions=True #Boolean variable to test if all predictions are correct for the current epoch
        
        for i in range(len(t)):

            #TODO - Application 1 - Step 4 - Call the forwardPropagation method
            a = forwardPropagation(P[i], weights, bias)

            #TODO - Application 1 - Step 5 - Compute the prediction error (error)
            error = t[i] - a

            #TODO - Application 1 - Step 6 - Update the weights
            weights[0] += error * P[i][0]
            weights[1] += error * P[i][1]    
            #TODO - Application 1 - Step 7 - Update the bias
            bias += error

            #Test if all the predictions of this epoch are correct or not 
            if error != 0:
                valid_predictions = False

        #If all predictions are correct
        if valid_predictions:
            print(f"Model converged after {ep} epochs.")
            break

    #TODO - Application 1 - Step 8 - Print weights and bias
    print("Final weights:", weights)
    print("Final bias:", bias)
   
    # TODO - Application 1 - Step 9 - Display the results
    print("Input\t\tPredicted\tTarget")
    for i in range(len(t)):
        prediction = forwardPropagation(P[i], weights, bias)
        print(f"{P[i]}\t\t{prediction}\t\t{t[i]}")

   
    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == "__main__":
    main()
#####################################################################################################################
#####################################################################################################################