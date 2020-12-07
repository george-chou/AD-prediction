import os
from csv import reader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC

def norm(arr):
    x_max = max(arr)
    x_min = min(arr)
    for i in range(len(arr)):
        arr[i] = (arr[i] - x_min) / (x_max - x_min)
    return arr

# Load training data
with open("data/AAL_statistics_volumn_train.csv") as f:
    csv_data = reader(f, delimiter=',')
    raw_data = np.array(list(csv_data))

# Preprocess training data
x_train = []
y_train = []
data_count = len(raw_data)
tuple_len = len(raw_data[0])

for i in raw_data:
    temp = norm([int(j) for j in i[0:tuple_len - 2]])
    x_train.append(temp)
    if i[tuple_len - 1] == "yes":
        y_train.append(1)
    else:
        y_train.append(0)

# Load test data
with open("data/AAL_statistics_volumn_test.csv") as f:
    csv_data = reader(f, delimiter=',')
    raw_data = np.array(list(csv_data))

# Preprocess test data
x_test = []
y_test = []
data_count = len(raw_data)
tuple_len = len(raw_data[0])

for i in raw_data:
    temp = norm([int(j) for j in i[0:tuple_len - 1]])
    x_test.append(temp)

#predict
clf = LinearSVC(loss="hinge", random_state=42).fit(x_train, y_train)
y_test = clf.predict(x_test)
print(y_test)

# Split dataset
xt_train, xt_test, yt_train, yt_test = train_test_split(x_train, y_train, test_size=0.33, random_state=73)

#accuracy
clf = LinearSVC(loss="hinge", random_state=42).fit(xt_train, yt_train)
print(clf.score(xt_test, yt_test))