import numpy as np
import matplotlib.pyplot as plt

# Define the Mel scale conversion function
def hertz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

# Define the frequency range (Hertz)
hertz = np.linspace(0, 10000, 1000)  # From 0 to 10,000 Hz

# Convert Hertz to Mel scale
mel = hertz_to_mel(hertz)

# Plot the Hertz scale vs Mel scale
plt.figure(figsize=(10, 5))
plt.plot(hertz, mel, color='red')

# Add labels and title
plt.xlabel('Hertz scale')
plt.ylabel('Mel scale')
plt.title('Hertz to Mel Scale Conversion')
plt.grid(True)

# Display the plot
plt.show()
