# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    describe_data_train.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/19 13:08:56 by msukhare          #+#    #+#              #
#    Updated: 2018/10/27 14:58:02 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from read_and_complete_data import read_file
import sys

def get_tab_split(cols_data, Y):
    yes = []
    no = []
    for i in range(cols_data.shape[0]):
        if (pd.isna(Y[i])):
            break
        if (Y[i] == 1):
            yes.append(cols_data[i])
        else:
            no.append(cols_data[i])
    return (yes, no)

def histo_data_train(data):
    nb_cols = 3
    nb_fig = 6
    fig, axs = plt.subplots(nb_cols, nb_fig, figsize=(7, 7))
    i = 0
    j = 0
    for key in data:
        if (key != "PassengerId" and key != "Survived"):
            if (j == nb_fig):
                i += 1
                j = 0
            yes, no = get_tab_split(data[key], data['Survived'])
            axs[i, j].hist([yes, no], bins="auto", color=['green', 'black'],\
            label=['yes', 'no'], histtype='bar')
            axs[i, j].legend()
            axs[i, j].set_title(key)
            j += 1
    plt.show()

def main():
    data_train, old_data = read_file(1)
    #for i in range(int(data.shape[0])):
        #if (pd.notna(data['Cabin'][i])):
            #print(data['Cabin'][i], data['Ticket'][i], data['Fare'][i])
    histo_data_train(data_train)
    tmp = old_data.head(891)
    tmp = tmp.corr()
    sns.heatmap(tmp, xticklabels=tmp.columns.values, yticklabels=tmp.columns.values)
    plt.show()
    tmp = data_train.head(891)
    tmp = tmp.corr()
    sns.heatmap(tmp, xticklabels=tmp.columns.values, yticklabels=tmp.columns.values)
    plt.show()

if (__name__ == "__main__"):
    main()
