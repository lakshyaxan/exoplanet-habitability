import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("ESI-3.csv", sep=',')
dataset = dataset.dropna()

X = dataset.iloc[:, 2:-1].values
Y = dataset.iloc[:, -1].values

# Hyperparameters
input_size = X.shape[1]
hidden_size = 10
output_size = 1
epochs = 50000
learning_rate = 0.000001  # Lower learning rate
num_hidden_layers = 5
alpha = 0.01  # Slope of negative part for Leaky ReLU

# Weight and bias initialization (using He initialization for hidden layers)
W = []
b = []

W.append(np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size))  # He initialization for first layer
b.append(np.zeros((1, hidden_size)))

for _ in range(num_hidden_layers - 1):
    W.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size))  # He initialization for hidden layers
    b.append(np.zeros((1, hidden_size)))

W.append(np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size))  # Output layer
b.append(np.zeros((1, output_size)))


# Leaky ReLU activation function
def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)  # if Z > 0 return Z, else return alpha * Z

# Derivative of Leaky ReLU
def leaky_relu_derivative(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)  # Derivative: 1 if Z > 0, else alpha


def forward_propagation(X, W, b):
    A = []
    Z = np.dot(X, W[0]) + b[0]
    A.append(leaky_relu(Z))  # Leaky ReLU activation
    for i in range(1, num_hidden_layers):
        Z = np.dot(A[i - 1], W[i]) + b[i]
        A.append(leaky_relu(Z))
    Z_output = np.dot(A[-1], W[-1]) + b[-1]
    A.append(Z_output)
    return A


def backward_propagation(X, Y, A, W):
    m = X.shape[0]
    dZ = A[-1] - Y.reshape(-1, 1)
    dW = [1 / m * np.dot(A[i - 1].T, dZ) if i != 0 else 1 / m * np.dot(X.T, dZ) for i in range(num_hidden_layers + 1)]
    db = [1 / m * np.sum(dZ, axis=0, keepdims=True) for _ in range(num_hidden_layers + 1)]

    for i in range(num_hidden_layers - 1, -1, -1):
        dA = np.dot(dZ, W[i + 1].T)
        dZ = dA * leaky_relu_derivative(A[i])  # Using Leaky ReLU derivative
        if i > 0:
            dW[i] = 1 / m * np.dot(A[i - 1].T, dZ)
        else:
            dW[i] = 1 / m * np.dot(X.T, dZ)
        db[i] = 1 / m * np.sum(dZ, axis=0, keepdims=True)

    return dW, db


# Update weights and biases using gradient descent
def update_parameters(W, b, dW, db, learning_rate):
    for i in range(num_hidden_layers + 1):
        W[i] -= learning_rate * dW[i]
        b[i] -= learning_rate * db[i]
    return W, b


# Training the model
for epoch in range(epochs):
    A = forward_propagation(X, W, b)

    # Backward propagation
    dW, db = backward_propagation(X, Y, A, W)

    # Update parameters
    W, b = update_parameters(W, b, dW, db, learning_rate)

    # Print the loss (Mean Squared Error) for every 1000 epochs to track progress
    if epoch % 1000 == 0:
        loss = np.mean((A[-1] - Y.reshape(-1, 1)) ** 2)
        print(f"Epoch {epoch}, Loss: {loss}")


# Making predictions
def predict(X, W, b):
    A = forward_propagation(X, W, b)
    return A[-1]


# Example of making a prediction for a sample input
sample_input = X[8].reshape(1, -1)  
predicted_value = predict(sample_input, W, b)
print("Predicted value:", predicted_value[0], Y[8])

# Multiple parameters optimisation algorithm

input_file = "habitability_scored_data.csv"
df = pd.read_csv(input_file,sep=';')

p_esi_column = 'P_ESI'
habitability_score_column = 'habitability_score'

p_esi_data = df[p_esi_column].values
habitability_score_data = df[habitability_score_column].values

def compute_averages(data, chunk_size=50):
    num_chunks = len(data) // chunk_size
    averaged_values = []
    for i in range(num_chunks):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        averaged_values.append(np.mean(chunk))
    return averaged_values

averaged_p_esi = compute_averages(p_esi_data)
averaged_habitability_score = compute_averages(habitability_score_data)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.bar(range(len(averaged_p_esi)), averaged_p_esi, color='lightblue')
plt.xlabel('Chunk Number')
plt.ylabel('Average P_ESI')
plt.title('Average P_ESI of Every 50 Data Points')
plt.xticks(range(len(averaged_p_esi)), [f'Chunk {i+1}' for i in range(len(averaged_p_esi))])

plt.subplot(2, 1, 2)
plt.bar(range(len(averaged_habitability_score)), averaged_habitability_score, color='lightgreen')
plt.xlabel('Chunk Number')
plt.ylabel('Average Habitability Score')
plt.title('Average Habitability Score of Every 50 Data Points')
plt.xticks(range(len(averaged_habitability_score)), [f'Chunk {i+1}' for i in range(len(averaged_habitability_score))])

plt.tight_layout()
plt.show()