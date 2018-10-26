# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    read_and_complete_data.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/22 16:33:56 by msukhare          #+#    #+#              #
#    Updated: 2018/10/26 17:44:37 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import sys
import time

def create_feat_size_family(sibsp, parch):
    new_feat = np.zeros((sibsp.shape[0], 1), dtype=int)
    for i in range(int(sibsp.shape[0])):
        new_feat[i][0] = sibsp[i] + parch[i]
    return (new_feat)

def categorize_age(age):
    new_feat = np.zeros((age.shape[0], 1), dtype=int)
    for i in range(int(age.shape[0])):
        if (age[i] <= 16):
            new_feat[i][0] = 1
    return (new_feat)

def dummy_feat(embarked, letter):
    new_feat = np.zeros((embarked.shape[0], 1), dtype=int)
    for i in range(int(embarked.shape[0])):
        if (pd.notna(embarked[i]) and embarked[i] == letter):
            new_feat[i][0] = 1
    return (new_feat)

def complete_data(data):
    data['Sex'] = data['Sex'].map({'male' : 1, 'female': 0})
    data.fillna(value={'Age': data['Age'].mean()}, inplace=True)
    """
    data.insert(data.shape[1], 'S', dummy_feat(data['Embarked'], 'S'))
    data.insert(data.shape[1], 'C', dummy_feat(data['Embarked'], 'C'))
    data.insert(data.shape[1], 'Q', dummy_feat(data['Embarked'], 'Q'))
    """
    #data.insert(data.shape[1], 'Child_or_not', categorize_age(data['Age']))
    data.insert(data.shape[1], 'Size_family', \
            create_feat_size_family(data['SibSp'], data['Parch']))
    data = data.drop(['Name', 'Ticket', 'Cabin', 'Parch', 'SibSp', 'Age', 'Fare', 'Embarked', 'Size_family'], axis=1)
    return (data)

def read_file(data_or_not):
    try:
        data_train = pd.read_csv("train.csv")
    except:
        sys.exit("fail to open train.csv")
    try:
        data_test = pd.read_csv("test.csv")
        tmp = np.full([data_test.shape[0], 1], np.nan)
        data_test.insert(1, 'Survived', tmp)
    except:
        sys.exit("fail to open test.csv")
    data = [data_train, data_test]
    data = pd.concat(data).reset_index()
    data = data.drop(['index'], axis=1)
    old_data = data
    data = complete_data(data)
    if (data_or_not == 1):
        return (data, old_data)
    passager_id = data.iloc[891: , 0: 1]
    passager_id = np.array(passager_id.values, dtype=int)
    Y_train = data.iloc[0: 891, 1: 2]
    Y_train = np.array(Y_train.values, dtype=int)
    X_train = data.iloc[0: 891, 2: ]
    X_train = np.array(X_train.values, dtype=float)
    X_test = data.iloc[891: , 2: ]
    X_test = np.array(X_test.values, dtype=float)
    return (X_train, Y_train, X_test, passager_id)
