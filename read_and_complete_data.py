# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    read_and_complete_data.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/22 16:33:56 by msukhare          #+#    #+#              #
#    Updated: 2018/10/23 18:04:43 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import sys

def create_feat_size_family(sibsp, parch):
    new_feat = np.zeros((sibsp.shape[0], 1), dtype=int)
    for i in range(int(sibsp.shape[0])):
        new_feat[i][0] = sibsp[i] + parch[i]
    return (new_feat)

def if_child(data):
    return (0)

def change_Age_into_bool(data):
    new_feat = np.zeros((data.shape[0], 1), dtype=int)
    for i in range(int(data.shape[0])):
        """
        if (pd.isna(data['Age'][i])):
            new_feat[i][0] = if_child(data)
            """
        if (data['Age'][i] < 17):
            new_feat[i][0] = 1
    return (new_feat)

def categorize_age(age):
    new_feat = np.zeros((age.shape[0], 1), dtype=int)
    for i in range(int(age.shape[0])):
        if (age[i] >= 0 and age[i] <= 16):
            new_feat[i][0] = 1
        elif (age[i] > 16 and age[i] <= 40):
            new_feat[i][0] = 2
        else:
            new_feat[i][0] = 3
    return (new_feat)

def check_if_same_data(name, ticket, sibsp, sex, pos):
    for i in range(name.shape[0]):
        if (i != pos and ticket[i] == ticket[pos] and sibsp[i] > 0):
            if (sex[pos] == 1):
                last_name_find = name[pos].split(",")
                last_name_find = last_name_find[1]
            else:
                last_name_find = name[pos].split("(")
                last_name_find = last_name_find[0].split(",")
                last_name_find = last_name_find[1]
            if (sex[i] == 1):
                last_name = name[i].split(",")
                last_name = last_name[1]
            else:
                last_name = name[i].split("(")
                last_name = last_name[0].split(",")
                last_name = last_name[1]
            if (last_name == last_name_find):
                return (1)
    return (0)

def create_married(data):
    new_feat = np.zeros((data.shape[0], 1), dtype=int)
    for i in range(data.shape[0]):
        if (data['SibSp'][i] > 0 and check_if_same_data(data['Name'], data['Ticket'],\
                data['SibSp'], data['Sex'], i)):
            new_feat[i][0] = 1
    return (new_feat)

def complete_data(data):
    data['Sex'] = data['Sex'].map({'male' : 1, 'female': 0})
    data['Embarked'] = data['Embarked'].map({'S' : 1, 'C': 2, 'Q': 3})
    data.fillna(value={'Age': data['Age'].quantile(0.50)}, inplace=True)
    data.fillna(value={'Embarked': data['Embarked'].quantile(0.50)}, inplace=True)
    data.fillna(value={'Fare': data['Fare'].quantile(0.50)}, inplace=True)
    data.insert(data.shape[1], 'Child_or_not', change_Age_into_bool(data))
    #data.insert(data.shape[1], 'Age_category', categorize_age(data['Age']))
    data.insert(data.shape[1], 'Married', create_married(data))
    data.insert(data.shape[1], 'Size_family', \
            create_feat_size_family(data['SibSp'], data['Parch']))
    data = data.drop(['Name', 'Ticket', 'Cabin', 'Parch', 'SibSp', 'Age'], axis=1)
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
    """
    for i in range(int(data.shape[0])):
        if (pd.isna(data['Age'][i])):
            print(data['Name'][i], data['Parch'][i], data['SibSp'][i], data['Ticket'][i], data['Fare'][i], data['Cabin'][i], data['Pclass'][i], data['Age'][i])
    sys.exit()
    """
    data = complete_data(data)
    if (data_or_not == 1):
        return (data)
    passager_id = data.iloc[891: , 0: 1]
    passager_id = np.array(passager_id.values, dtype=int)
    Y_train = data.iloc[0: 891, 1: 2]
    Y_train = np.array(Y_train.values, dtype=int)
    X_train = data.iloc[0: 891, 2: ]
    X_train = np.array(X_train.values, dtype=float)
    X_test = data.iloc[891: , 2: ]
    X_test = np.array(X_test.values, dtype=float)
    return (X_train, Y_train, X_test, passager_id)
