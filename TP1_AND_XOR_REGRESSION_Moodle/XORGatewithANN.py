import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Define the XOR gate dataset as it has been defined in the lab
def define_dataset():
    X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)
    return X, y

# Create the ANN model architecture
#Here we have used two layers to allow for a more complex representation of the xor gate problem
#with the XOR layer, we need a non linear decision boundary 

def model(input_dim):
    model = Sequential([
        #the first hidden layer extracts relevant features from the input data, 
        Dense(2, activation='sigmoid', input_dim=input_dim),
        #and the second hidden layer combines these features to make the final prediction.  
        Dense(1, activation='sigmoid')
    ])
    return model

# Train the ANN model with the same implementation as the ANDgate
def train_model(model, X, y, optimizer='adam', loss='binary_crossentropy', epochs=1000):
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 
    model.fit(X, y, epochs=epochs)
    return model

# Evaluate the model on the XOR gate dataset using accuracy and loss
def evaluate_model(model, X, y):
    loss, accuracy = model.evaluate(X, y)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

# Main function
def main():
    # Define the XOR gate dataset extractinf the input and the output
    X_xor, y_xor = define_dataset()

    # Create the ANN model
    model_xor = model(input_dim=2)

    # Train the model
    print("Training the model for the XOR gate...")
    model_xor = train_model(model_xor, X_xor, y_xor)

    # Evaluate the model
    print("Evaluation for the XOR gate:")
    loss, accuracy = model_xor.evaluate(X_xor, y_xor)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
