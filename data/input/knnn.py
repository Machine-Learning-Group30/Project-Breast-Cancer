
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
df.head()

df.index = df['id']
df.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
df.head()

#subset into train and test set
diagnosis = df.diagnosis
df.drop(['diagnosis'], axis = 1, inplace = True)

from sklearn.cross_validation import train_test_split as tspl
df_train, df_test, diag_train, diag_test = tspl(df, diagnosis, test_size = 0.33)

# KNN Classifier

from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors = 5)
knn.fit(df_train, diag_train)

print(knn.score(df_test, diag_test))