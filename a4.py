import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X_train = np.random.uniform(1, 10, 20)
Y_train = np.random.uniform(1, 10, 20)
#if X+Y>10 then class 1 = red else class 0 = blue
train_classes = np.where(X_train + Y_train > 10, 1, 0)
train_data = np.column_stack((X_train, Y_train))

#mesh grid of X and Y from 0 to 10 with 0.1 step
x_vals = np.arange(0, 10.1, 0.1)
y_vals = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_vals, y_vals)
test_data = np.column_stack((X_test.ravel(), Y_test.ravel()))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_classes)
predicted = knn.predict(test_data)

# Assign colors based on predicted class
colors = []
for c in predicted:
    if c == 0:
        colors.append('blue')
    else:
        colors.append('red')

#plot
plt.scatter(test_data[:, 0], test_data[:, 1], c=colors, s=1)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Test Data Classification")
plt.show()
