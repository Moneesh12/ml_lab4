import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("C:\\Users\\jeeva\\Desktop\\ML_assignment4\\DCT_mal.csv")
df = df[df['LABEL'].isin([1, 2])]
df = df[['1', '2', 'LABEL']]
# select 20 random rows
df = df.sample(20, random_state=0)

x = df[['1', '2']].values
y = df['LABEL'].values

# Using GridSearchCV operations to find the ideal 'k'
param = {'n_neighbors': list(range(1, 5))}
model = KNeighborsClassifier()
grid = GridSearchCV(model, param, cv=5)
grid.fit(x, y)

print("Best k value using Gridsearch:", grid.best_params_['n_neighbors'])

#Using RandomizedSearch operation to find ideal 'k'
param = {'n_neighbors': randint(1, 11)}
model = KNeighborsClassifier()
random_search = RandomizedSearchCV(model, param, n_iter=10, cv=5, random_state=0)
random_search.fit(x, y)

print("Best k using RandomizedSearchCV:", random_search.best_params_['n_neighbors'])
