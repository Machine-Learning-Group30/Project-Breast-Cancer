
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

# for i in range(1, 20):
#     knn = KNN(n_neighbors = i)
#     knn.fit(df_train, diag_train)
#     score = knn.score(df_test, diag_test)
#     print("N = " + str(i) + " Score = " + str(score))
# ideal number of neighbors = 7

knn = KNN(n_neighbors = 7)
knn.fit(df_train, diag_train)

pred_prob = knn.predict_proba(df_test)
predictions = knn.predict(df_test)

from sklearn.metrics import confusion_matrix as cfm

print(cfm(predictions, diag_test))

# confusion table:
# [[109  10]
#  [  3  66]]