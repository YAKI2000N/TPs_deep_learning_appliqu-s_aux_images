import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Function to define the AND gate dataset
#we have tried with the same dataset provided in the lab 
def define_dataset():
    X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y = tf.constant([[0], [0], [0], [1]], dtype=tf.float32)
    return X, y

# Function to create the ANN model for the AND gate
#we need a a fully connected layer with one neuron because the output is binary. 
def model(input_dim): #the dimension of the input data is 2 (0 or 1)
    model = Sequential([
        Dense(1, activation='sigmoid', input_dim=input_dim) #The sigmoid function squashes the output to a range between 0 and 1
    ])
    return model

# Function to train the ANN model
#here the binary_cross_entropy is used for a binary classification instead of crossentropy function
def train_model(model, X, y, optimizer='adam', loss='binary_crossentropy', epochs=100):
    #configuring the model for training 
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) #we use accuracy to test the performance of training
    #trains the NN using the provided input data X and the target labels Y 
    model.fit(X, y, epochs=epochs)
    return model

# Main function
def main():
    # Train the model for the AND gate
    print("Training the model for the AND gate...")
    X_and, y_and = define_dataset()
    model_and = model(input_dim=2)
    model_and = train_model(model_and, X_and, y_and)
    print("Model trained for the AND gate.")

    # Evaluate the model for the AND gate
    #The evaluate method returns a list of two values: the loss value and the value of the metrics specified during model compilation (in this case, accuracy)
    #The evaluate function is a method available in the keras "sequential" model
    print("Evaluation for the AND gate:")
    loss, accuracy = model_and.evaluate(X_and, y_and)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
