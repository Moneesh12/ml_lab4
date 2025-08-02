import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"C:\\Users\\jeeva\\Desktop\\ML_assignment4\\DCT_mal.csv")
df = df[df['LABEL'].isin([1, 2])]
df = df[['0', '1', 'LABEL']]
df = df.sample(20, random_state=0)

x = df[['0', '1']].values
y = df['LABEL'].values

#assign color to point based on its label
colors = ['blue' if i == 1 else 'red' for i in y]
plt.figure(figsize=(6, 4))
plt.scatter(x[:, 0], x[:, 1], c=colors, marker='o', s=50, edgecolor='black')
plt.title('Training Data')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# grid for plot
a = np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 100)
b = np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 100)
xx, yy = np.meshgrid(a, b)
test = np.column_stack((xx.ravel(), yy.ravel()))


#we are looping for different k values and we will get diffferent k graphs
for k in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x, y)
    pred = model.predict(test)

    plt.figure(figsize=(6, 4))
    sampled_idx = np.random.choice(len(test), size=len(test) // 10, replace=False)
    # Plot predicted class 1,2 and actual training data for class 1 and 2
    plt.scatter(test[sampled_idx][pred[sampled_idx] == 1][:, 0], test[sampled_idx][pred[sampled_idx] == 1][:, 1], c='blue', s=5, alpha=0.3)
    plt.scatter(test[sampled_idx][pred[sampled_idx] == 2][:, 0], test[sampled_idx][pred[sampled_idx] == 2][:, 1], c='red', s=5, alpha=0.3)
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c='blue', marker='o', s=50, edgecolor='black')
    plt.scatter(x[y == 2][:, 0], x[y == 2][:, 1], c='red', marker='o', s=50, edgecolor='black')
    plt.title(f'k = {k}')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.tight_layout()
    plt.show()
