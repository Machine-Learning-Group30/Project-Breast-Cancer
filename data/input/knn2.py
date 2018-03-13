import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter

#column names
names = ['id','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean', 'concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se'
,'texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst'
 ,'smoothness_worst','compactness_worst','concavity_worst' ,'concave_points_worst','symmetry_worst','fractal_dimension_worst', 'diagnosis']

#load training set
df = pd.read_csv('normalized_trainingdata.csv', header=None, names=names)

X = np.array(df.ix[:, 0:])
y = np.array(df['diagnosis'])

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# learning model (k = 3)
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# pred = knn.predict(X_test)
# accuracy = score_accuracy(y_test, pred) * 100
#
# print('The accuracy of the knn classifier for k = 3 is %d%%' % accuracy)

def train(X_train, y_train):
    return

def predict(X_train, y_train, X_test, k):
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute euclidean distance
        distance = np.sqrt(np.sum(np.square(X_test - X_train[i, :])))
        distances.append([distance, i])

    distances = sorted(distances)

    for j in range(k):
        index = distances[j][1]
        targets.append(y_train[index])

    return Counter(targets).most_common(1)[0][0]

def knn(X_train, y_train, X_test, predictions, k):
    if k > len(X_train):
        raise ValueError

    train(X_train, y_train)

    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))


predictions = []
try:
    knn(X_train, y_train, X_test, predictions, 7)
    predictions = np.asarray(predictions)

    accuracy = accuracy_score(y_test, predictions) * 100
    print('The accuracy of our classifier is', accuracy)

except ValueError:
    print('Can not have more neighbors than training samples')