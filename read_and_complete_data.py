# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    read_and_complete_data.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/22 16:33:56 by msukhare          #+#    #+#              #
#    Updated: 2018/10/28 11:21:20 by kemar            ###   ########.fr        #
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
            """
        elif (age[i] <= 32):
            new_feat[i][0] = 1
        elif (age[i] <= 34):
            new_feat[i][0] = 2
        elif (age[i] <= 48):
            new_feat[i][0] = 3
        elif (age[i] <= 51):
            new_feat[i][0] = 4
        elif (age[i] <= 63):
            new_feat[i][0] = 5
        else:
            new_feat[i][0] = 6
            """
    return (new_feat)

def dummy_feat(embarked, letter):
    new_feat = np.zeros((embarked.shape[0], 1), dtype=int)
    for i in range(int(embarked.shape[0])):
        if (pd.notna(embarked[i]) and embarked[i] == letter):
            new_feat[i][0] = 1
    return (new_feat)

def create_fare_person(size_family, fare):
    new_feat = np.zeros((fare.shape[0], 1), dtype=float)
    for i in range(fare.shape[0]):
        new_feat[i][0] = fare[i] / (size_family[i] + 1)
    return (new_feat)

def create_feat_title(name):
    new_feat = np.zeros((name.shape[0], 1), dtype=int)
    for i in range(name.shape[0]):
        title = name[i].split(",")
        title = title[1].split(".")
        title = title[0].strip()
        if (title == "Mr" or title == "Ms" or title == "Mrs" or title == "Mme"):
            new_feat[i][0] = 0
        elif (title == "Miss" or title == "Mlle" or title == "Master"):
            new_feat[i][0] = 1
        else:
            new_feat[i][0] = 2
            """
        elif (title == "Jonkheer" or title == "Don" or title == "Sir" or title == "Dona" or\
                title == "the Countess" or title == "Lady"):
            new_feat[i][0] = 2
        else:
            new_feat[i][0] = 3
            """
    return (new_feat)

def complete_data(data):
    data['Sex'] = data['Sex'].map({'male' : 1, 'female': 0})
    data['Embarked'] = data['Embarked'].map({'S' : 1, 'C': 2, 'Q': 3})
    data.fillna(value={'Age': data['Age'].mean()}, inplace=True)
    data.fillna(value={'Fare': data['Fare'].mean()}, inplace=True)
    data.fillna(value={'Embarked': data['Embarked'].quantile(0.50)}, inplace=True)
    data.insert(data.shape[1], 'S', dummy_feat(data['Embarked'], 1))
    data.insert(data.shape[1], 'C', dummy_feat(data['Embarked'], 2))
    data.insert(data.shape[1], 'Q', dummy_feat(data['Embarked'], 3))
    data.insert(data.shape[1], 'Child_or_not', categorize_age(data['Age']))
    data.insert(data.shape[1], 'Size_family', \
            create_feat_size_family(data['SibSp'], data['Parch']))
    data.insert(data.shape[1], 'Fare per person', create_fare_person(data['Size_family'], data['Fare']))
    data.insert(data.shape[1], 'Title', create_feat_title(data['Name']))
    data.insert(data.shape[1], 'Mr_Or_Ms', dummy_feat(data['Title'], 0))
    data.insert(data.shape[1], 'Miss_Or_Master', dummy_feat(data['Title'], 1))
    data.insert(data.shape[1], 'Royal_and_Officer', dummy_feat(data['Title'], 2))
    data = data.drop(['Parch', 'SibSp', 'Age', 'Fare', 'Embarked', 'Name',\
            'Ticket', 'Cabin', 'Title'], axis=1)
    return (data)

def read_file(ret_with_old):
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
    data = data.drop(['index', 'PassengerId'], axis=1)
    data_old = data
    data = complete_data(data)
    data_train = data.iloc[0: 891]
    data_test = data.iloc[891:]
    data_test = data_test.reset_index()
    data_test = data_test.drop(['index', 'Survived'], axis=1)
    if (ret_with_old == 1):
        return (data_train, data_old)
    return (data_train, data_test)
