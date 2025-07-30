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

# Training predictions
y_train_pred = neigh.predict(X_train)
conf_train = confusion_matrix(y_train, y_train_pred)
report_train = classification_report(y_train, y_train_pred)
print("Training Confusion Matrix:\n", conf_train)
print("Training Classification Report:\n", report_train)

# Testing predictions
y_test_pred = neigh.predict(X_test)
conf_test = confusion_matrix(y_test, y_test_pred)
report_test = classification_report(y_test, y_test_pred)
print("Test Confusion Matrix:\n", conf_test)
print("Test Classification Report:\n", report_test)

train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

if train_accuracy > 0.95 and test_accuracy < 0.85:
    print("Inference: Model is Overfitting.")
elif abs(train_accuracy - test_accuracy) <= 0.05:
    print("Inference: Model is Regular fit.")
else:
    print("Inference: Model may be Underfitting.")
