import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
# Converting 'Volume' column to float
df['Volume'] = df['Volume'].str.replace('K', 'e3').str.replace('M', 'e6').astype(float)

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Price']

#linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("MAPE:", mean_absolute_percentage_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
