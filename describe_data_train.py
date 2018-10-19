# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    describe_data_train.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/19 13:08:56 by msukhare          #+#    #+#              #
#    Updated: 2018/10/19 17:00:21 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_tab_split(cols_data, data):
    yes = []
    no = []
    for i in range(cols_data.shape[0]):
        if (data['Survived'][i] == 1):
            yes.append(cols_data[i])
        else:
            no.append(cols_data[i])
    return (yes, no)

def complete_data(data_train):
    data_train['Sex'] = data_train['Sex'].map({'male' : 1, 'female': 0})
    data_train['Embarked'] = data_train['Embarked'].map({'S' : 1, 'C': 2, 'Q': 3})
    data_train.fillna(value={'Age': data_train['Age'].quantile(0.50)}, inplace=True)
    data_train.fillna(value={'Embarked': data_train['Embarked'].quantile(0.50)}, inplace=True)
    return (data_train)

def histo_data_train(data_train):
    fig, axs = plt.subplots(2, 4, figsize=(7, 7))
    yes, no = get_tab_split(data_train['Pclass'], data_train)
    axs[0, 0].hist(yes, bins="auto", facecolor='green', label='yes')
    axs[0, 0].hist(no, bins="auto", facecolor='black', alpha=0.6, label='no')
    axs[0, 0].legend()
    axs[0, 0].set_title("Pclass")
    yes, no = get_tab_split(data_train['Sex'], data_train)
    axs[0, 1].hist(yes, bins="auto", facecolor='green', label='yes')
    axs[0, 1].hist(no, bins="auto", facecolor='black', alpha=0.6, label='no')
    axs[0, 1].legend()
    axs[0, 1].set_title("Sex")
    yes, no = get_tab_split(data_train['Age'], data_train)
    axs[0, 2].hist(yes, bins="auto", facecolor='green', label='yes')
    axs[0, 2].hist(no, bins="auto", facecolor='black', alpha=0.6, label='no')
    axs[0, 2].legend()
    axs[0, 2].set_title("Age")
    yes, no = get_tab_split(data_train['SibSp'], data_train)
    axs[0, 3].hist(yes, bins="auto", facecolor='green', label='yes')
    axs[0, 3].hist(no, bins="auto", facecolor='black', alpha=0.6, label='no')
    axs[0, 3].legend()
    axs[0, 3].set_title("SibSp")
    yes, no = get_tab_split(data_train['Parch'], data_train)
    axs[1, 0].hist(yes, bins="auto", facecolor='green', label='yes')
    axs[1, 0].hist(no, bins="auto", facecolor='black', alpha=0.6, label='no')
    axs[1, 0].legend()
    axs[1, 0].set_title("Parch")
    yes, no = get_tab_split(data_train['Fare'], data_train)
    axs[1, 1].hist(yes, bins="auto", facecolor='green', label='yes')
    axs[1, 1].hist(no, bins="auto", facecolor='black', alpha=0.6, label='no')
    axs[1, 1].legend()
    axs[1, 1].set_title("Fare")
    yes, no = get_tab_split(data_train['Embarked'], data_train)
    axs[1, 2].hist(yes, bins="auto", facecolor='green', label='yes')
    axs[1, 2].hist(no, bins="auto", facecolor='black', alpha=0.6, label='no')
    axs[1, 2].legend()
    axs[1, 2].set_title("Embarked")
    plt.show()

def main():
    try:
        data_train = pd.read_csv("train.csv")
    except:
        sys.exit("fail to open file")
    print(data_train)
    print(data_train.describe())
    data_modif = complete_data(data_train)
    #for i in range(int(data_train['Cabin'].shape[0])):
        #print(data_train['Cabin'][i], "\t", data_train['Name'][i], "\t", data_train['Survived'][i])
    histo_data_train(data_modif)


if (__name__ == "__main__"):
    main()
