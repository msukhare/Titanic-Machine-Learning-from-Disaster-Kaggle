# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_rand_tree.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/22 16:14:31 by msukhare          #+#    #+#              #
#    Updated: 2018/10/26 17:44:36 by msukhare         ###   ########.fr        #
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

def main():
    try:
        Y_test = pd.read_csv("gender_submission.csv")
    except:
        sys.exit("fail to open file")
    Y_test = Y_test.iloc[:, 1:]
    Y_test = np.array(Y_test.values, dtype=int)
    X, Y, X_test, passager_id = read_file(0)
    Y = np.reshape(Y, (Y.shape[0]))
    error_rate = []
    nb_tree = []
    """
    for i in range(1, 100): #40
        random_forest = RandomForestClassifier(n_estimators=i, random_state=97)
        random_forest.fit(X, Y)
        y_pred_train = random_forest.predict(X)
        error_rate.append(1 - accuracy_score(Y, y_pred_train))
        nb_tree.append(i)
        y_pred_test = random_forest.predict(X_test)
        print(accuracy_score(Y_test, y_pred_test))
    plt.plot(nb_tree, error_rate)
    plt.show()
    """
    random_forest = RandomForestClassifier(n_estimators=2, random_state=97)
    random_forest.fit(X, Y)
    y_pred_test = random_forest.predict(X_test)
    print(accuracy_score(Y_test, y_pred_test))
    with open('My_submisson.csv', mode="w") as file_to_write:
        file_to_write = csv.writer(file_to_write, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        file_to_write.writerow(["PassengerId", "Survived"])
        for i in range(y_pred_test.shape[0]):
            file_to_write.writerow([passager_id[i][0], y_pred_test[i]])

if (__name__ == "__main__"):
    main()
