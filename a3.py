import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate 20 random data points between 1 and 10 for X and Y
X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)

classes = np.where(X + Y > 10, 1, 0)

colors = []

#separating the points by color
for c in classes:
    if c == 0:
        colors.append('blue')
    else:
        colors.append('red')


# Step 4: Plotting the data
plt.figure(figsize=(8,6))
plt.scatter(X, Y, c=colors)
plt.title('Training Data Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
