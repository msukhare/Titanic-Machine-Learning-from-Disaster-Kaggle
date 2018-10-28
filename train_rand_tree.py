# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_rand_tree.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/22 16:14:31 by msukhare          #+#    #+#              #
#    Updated: 2018/10/28 11:19:10 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from describe_data_train import read_file
from sklearn.ensemble import RandomForestClassifier
from math import floor
from sklearn.metrics import accuracy_score

def split_X_train_and_test(X):
    X = X.sample(frac=1, random_state=999).reset_index(drop=True)
    X_train = X.iloc[0: floor(X.shape[0] * 0.8), 1: ]
    X_train = np.array(X_train.values, dtype=float)
    X_test = X.iloc[floor(X.shape[0] * 0.8): -1, 1: ]
    X_test = np.array(X_test.values, dtype=float)
    Y_train = X.iloc[0: floor(X.shape[0] * 0.8), 0: 1]
    Y_train = np.array(Y_train.values, dtype=int)
    Y_test = X.iloc[floor(X.shape[0] * 0.8): -1, 0: 1]
    Y_test = np.array(Y_test.values, dtype=int)
    return (X_train, np.reshape(Y_train, (Y_train.shape[0])), X_test, \
            np.reshape(Y_test, (Y_test.shape[0])))

def split_X_train(X):
    X_train = X.iloc[:, 1:]
    X_train = np.array(X_train.values, dtype=float)
    Y_train = X.iloc[:, 0: 1]
    Y_train = np.array(Y_train.values, dtype=float)
    return (X_train, np.reshape(Y_train, (Y_train.shape[0])))

def train_tree_and_test(X_train, Y_train, X_test, Y_test):
    error_rate = []
    nb_tree = []
    for i in range(1, 100): #40
        random_forest = RandomForestClassifier(n_estimators=i, random_state=37)
        random_forest.fit(X_train, Y_train)
        y_pred_train = random_forest.predict(X_train)
        error_rate.append(1 - accuracy_score(Y_train, y_pred_train))
        nb_tree.append(i)
        y_pred_test = random_forest.predict(X_test)
        print(accuracy_score(Y_test, y_pred_test))
    plt.plot(nb_tree, error_rate)
    plt.show()
    """
    random_forest = RandomForestClassifier(n_estimators=60, random_state=37)
    random_forest.fit(X_train, Y_train)
    #y_pred_test = random_forest.predict(X_test)
    #print(accuracy_score(Y_test, y_pred_test))
    """
    return (random_forest)

def write_prediction_in_file(random_forest, X_val, passenger_id):
    y_pred_val = random_forest.predict(X_val)
    print(y_pred_val.shape[0])
    with open('My_submisson.csv', mode="w") as file_to_write:
        file_to_write = csv.writer(file_to_write, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        file_to_write.writerow(["PassengerId", "Survived"])
        for i in range(y_pred_val.shape[0]):
            file_to_write.writerow([passenger_id[i][0], int(y_pred_val[i])])

def main():
    try:
        passenger_id = pd.read_csv("gender_submission.csv")
    except:
        sys.exit("fail to open file")
    passenger_id = passenger_id.iloc[:, 0:1]
    passenger_id = np.array(passenger_id.values, dtype=int)
    X_train, X_val  = read_file(0)
    #to test
    X_train, Y_train, X_test, Y_test = split_X_train_and_test(X_train)
    random_forest = train_tree_and_test(X_train, Y_train, X_test, Y_test)
    """#to predicted
    X_train, Y_train = split_X_train(X_train)
    random_forest = train_tree_and_test(X_train, Y_train, 0, 0)
    X_val = X_val.iloc[:]
    X_val = np.array(X_val.values, dtype=float)
    write_prediction_in_file(random_forest, X_val, passenger_id)
    """

if (__name__ == "__main__"):
    main()
