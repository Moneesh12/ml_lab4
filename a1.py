import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

rf = pd.read_csv("C:\\Users\\jeeva\\Desktop\\ML_assignment4\\DCT_mal.csv")

# we can select any classes
selected_classes = [1, 2]
rf_binary = rf[rf['LABEL'].isin(selected_classes)]

# separating features and label
x = rf_binary.drop('LABEL', axis=1)
y = rf_binary['LABEL']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#training
y1 = neigh.predict(X_train)
conf_mat = confusion_matrix(y_train,y1)
class_r = classification_report(y_train,y1)
print("the confusion matrix is\n:",conf_mat)
print("classification report:\n",class_r)

#testing

y2 = neigh.predict(X_test)
conf_mat1 = confusion_matrix(y_test,y2)
class_r1 = classification_report(y_test,y2)
print("the confusion matrix is\n:",conf_mat1)
print("classification report\n:",class_r1)

#accuracies
acc = neigh.score(X_test, y_test) 

acc1 = neigh.score(X_train,y_train)

print("accuracy for test set:",acc)

print("accuracy for training set:",acc1)

#checking if model is underfit overfit or regular fit
if acc > 0.95 and acc1 < 0.85:
    print("Inference: Model is Overfitting.")
elif abs(acc1 - acc) <= 0.05:
    print("Inference: Model is Regular fit.")
else:
    print("Inference: Model may be Underfitting.")
