import numpy as np
import matplotlib.pyplot as plt

# Define the Mel scale conversion function
def hertz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)


# # Define the frequency range (Hertz)
# hertz = np.linspace(0, 10000, 1000)  # From 0 to 10,000 Hz

# # Convert Hertz to Mel scale
# mel = hertz_to_mel(hertz)

# # Plot the Hertz scale vs Mel scale
# plt.figure(figsize=(10, 5))
# plt.plot(hertz, mel, color='red')
# # Add labels and title
# plt.xlabel('Hertz scale')
# plt.ylabel('Mel scale')
# plt.title('Hertz to Mel Scale Conversion')
# plt.grid(True)
# # Display the plot
# plt.show()

# ----------------------------------------------
# Plot activation functions

# Generate input data
x = np.linspace(-10, 10, 100)

# Plot Sigmoid
plt.figure(figsize=(6, 4))
plt.plot(x, sigmoid(x), label="Sigmoid", color="#72383D")
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# Plot ReLU
plt.figure(figsize=(6, 4))
plt.plot(x, relu(x), label="ReLU", color="#72383D")
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# Plot Tanh
plt.figure(figsize=(6, 4))
plt.plot(x, tanh(x), label="Tanh", color="#72383D")
plt.title("Tanh Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()