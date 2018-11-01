# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    read_and_complete_data.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/22 16:33:56 by msukhare          #+#    #+#              #
#    Updated: 2018/11/01 14:06:36 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import sys
import time

def create_feat_size_family(sibsp, parch):
    new_feat = np.zeros((sibsp.shape[0], 1), dtype=int)
    for i in range(int(sibsp.shape[0])):
        if ((sibsp[i] + parch[i]) <= 0):
            new_feat[i][0] = 0
        elif ((sibsp[i] + parch[i]) <= 5):
            new_feat[i][0] = 1
        else:
            new_feat[i][0] = 2
        #new_feat[i][0] = sibsp[i] + parch[i]
    return (new_feat)

def categorize_age(age):
    new_feat = np.zeros((age.shape[0], 1), dtype=int)
    for i in range(int(age.shape[0])):
        if (age[i] <= 14):
            new_feat[i][0] = 0
        elif (age[i] <= 32):
            new_feat[i][0] = 1
        elif (age[i] <= 48):
            new_feat[i][0] = 2
        elif (age[i] <= 64):
            new_feat[i][0] = 3
        else:
            new_feat[i][0] = 4
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
        if (title == "Mr" or title == "Ms"):
            new_feat[i][0] = 0
        elif (title == "Mrs" or title == "Mme"):
            new_feat[i][0] = 1
        elif (title == "Miss" or title == "Mlle"):
            new_feat[i][0] = 2
        elif (title == "Master"):
            new_feat[i][0] = 3
        else:
            new_feat[i][0] = 4
            """
        elif (title == "Jonkheer" or title == "Don" or title == "Sir" or title == "Dona" or\
                title == "the Countess" or title == "Lady"):
            new_feat[i][0] = 2
        else:
            new_feat[i][0] = 3
            """
    return (new_feat)

def get_feat_cabin(cabin):
    new_feat = np.zeros((cabin.shape[0], 1), dtype=int)
    for i in range(cabin.shape[0]):
        if (pd.notna(cabin[i])):
            new_feat[i][0] = 1
    return (new_feat)

def create_new_fare(fare):
    new_feat = np.zeros((fare.shape[0], 1), dtype=int)
    for i in range(fare.shape[0]):
        if (fare[i] <= 7.91):
            new_feat[i][0] = 0
        elif (fare[i] <= 14.454):
            new_feat[i][0] = 1
        elif (fare[i] <= 31):
            new_feat[i][0] = 2
        else:
            new_feat[i][0] = 3
    return (new_feat)

def complete_data(data):
    data['Sex'] = data['Sex'].map({'male' : 1, 'female': 0})
    data['Embarked'] = data['Embarked'].map({'S' : 1, 'C': 2, 'Q': 3})
    data.fillna(value={'Age': data['Age'].mean()}, inplace=True)
    data.fillna(value={'Fare': data['Fare'].mean()}, inplace=True)
    data.fillna(value={'Embarked': data['Embarked'].quantile(0.50)}, inplace=True)
    data.insert(data.shape[1], 'S', dummy_feat(data['Embarked'], 1))
    data.insert(data.shape[1], 'C', dummy_feat(data['Embarked'], 2))
    #data.insert(data.shape[1], 'Q', dummy_feat(data['Embarked'], 3))
    data.insert(data.shape[1], 'Categorize_age', categorize_age(data['Age']))
    data.insert(data.shape[1], 'Size_family', \
            create_feat_size_family(data['SibSp'], data['Parch']))
    #data.insert(data.shape[1], 'Fare per person', create_fare_person(data['Size_family'], data['Fare']))
    data.insert(data.shape[1], 'Categorize Fare', create_new_fare(data['Fare']))
    data.insert(data.shape[1], 'Title', create_feat_title(data['Name']))
    data.insert(data.shape[1], 'Has Cabin', get_feat_cabin(data['Cabin']))
    data = data.drop(['Parch', 'SibSp', 'Age', 'Fare', 'Embarked', 'Name',\
            'Ticket', 'Cabin'], axis=1)
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
    data = complete_data(data)
    data_train = data.iloc[0: 891]
    data_test = data.iloc[891:]
    data_test = data_test.reset_index()
    data_test = data_test.drop(['index', 'Survived'], axis=1)
    if (ret_with_old == 1):
        return (data_train)
    return (data_train, data_test)
