# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    describe_data_train.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/19 13:08:56 by msukhare          #+#    #+#              #
#    Updated: 2018/10/21 18:55:42 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_tab_split(cols_data, Y):
    yes = []
    no = []
    for i in range(cols_data.shape[0]):
        if (Y[i] == 1):
            yes.append(cols_data[i])
        else:
            no.append(cols_data[i])
    return (yes, no)

def create_feat_size_family(sibsp, parch):
    new_feat = np.zeros((sibsp.shape[0], 1), dtype=int)
    for i in range(int(sibsp.shape[0])):
        new_feat[i][0] = sibsp[i] + parch[i]
    return (new_feat)

def complete_data(data_train):
    data_train['Sex'] = data_train['Sex'].map({'male' : 1, 'female': 0})
    data_train['Embarked'] = data_train['Embarked'].map({'S' : 1, 'C': 2, 'Q': 3})
    data_train.fillna(value={'Age': data_train['Age'].quantile(0.50)}, inplace=True)
    data_train.fillna(value={'Embarked': data_train['Embarked'].quantile(0.50)}, inplace=True)
    data_train.insert(data_train.shape[1], 'Size_family', \
            create_feat_size_family(data_train['SibSp'], data_train['Parch']))
    return (data_train)

def histo_data_train(data_train):
    fig, axs = plt.subplots(2, 5, figsize=(7, 7))
    yes, no = get_tab_split(data_train['Pclass'], data_train['Survived'])
    axs[0, 0].hist([yes, no], bins="auto", color=['green', 'black'],\
            label=['yes', 'no'], histtype='bar')
    axs[0, 0].legend()
    axs[0, 0].set_title("Pclass")
    yes, no = get_tab_split(data_train['Size_family'], data_train['Survived'])
    axs[0, 1].hist([yes, no], bins="auto", color=['green', 'black'],\
            label=['yes', 'no'], histtype='bar')
    axs[0, 1].legend()
    axs[0, 1].set_title("Size_family")
    plt.show()

def main():
    try:
        data_train = pd.read_csv("train.csv")
    except:
        sys.exit("fail to open file")
    for i in range(int(data_train.shape[0])):
        tmp = data_train['Name'][i].split(",")
        for j in range(int(data_train.shape[0])):
            tmp1 = data_train['Name'][j].split(",")
            if (i != j and tmp[0] == tmp1[0]):
                print(data_train['Name'][j], data_train['Ticket'][j])
#    data_modif = complete_data(data_train)
 #   histo_data_train(data_modif)

if (__name__ == "__main__"):
    main()
