import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x = np.random.uniform(1, 10, 20)
y = np.random.uniform(1, 10, 20)
# Label as 1 if x + y > 10, else 0
label = np.where(x + y > 10, 1, 0)
train = np.column_stack((x, y))

# Create a grid of test points
a = np.arange(0, 10.1, 0.1)
b = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(a, b)
test = np.column_stack((xx.reshape(-1), yy.reshape(-1)))

#running it in a loop and also for getting graphs for k = 1,3,5,7 
for k in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train, label)
    pred = model.predict(test)
    arr = []
    for i in pred:
        if i == 0:
            arr.append('blue')
        else:
            arr.append('red')
#plot
    plt.scatter(test[:, 0], test[:, 1], c=arr, s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
