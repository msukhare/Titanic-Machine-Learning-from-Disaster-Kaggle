# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_rand_tree.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/22 16:14:31 by msukhare          #+#    #+#              #
#    Updated: 2018/10/22 18:17:45 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
from describe_data_train import read_file
from sklearn.ensemble import RandomForestClassifier

def main():
    X_train, Y_train, X_test, passager_id = read_file(0)
    for i in range(int(X_test.shape[0])):
        print(X_test[i])
    #random_forest = RandomForestClassifier(n_estimators=100)
    #random_forest.fit(X_train, Y_train)
    #y_pred = random_forest.predict(X_test)
   # print(Y_pred)

if (__name__ == "__main__"):
    main()
