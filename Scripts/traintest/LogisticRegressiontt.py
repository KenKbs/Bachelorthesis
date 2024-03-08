# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:31:16 2024

@author: Kenny
"""

from Scripts.util import (
    get_data,
    filter_data,
    generate_table,
    train_test_split_data)

from sklearn.linear_model import LogisticRegression

#Everything starts here

#%% read in

shading = True #put later into function

data=get_data()

#filter data
data=filter_data(data,filter_value=100,shading=shading)

#Print out fault distribution before and after filtering
pass
            

#%%Rpeaditly train_test data

#PUT INTO LOOP LATER!!!

# Split data w. own fuinction, scaling = True
x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                         test_size=0.2,scaling=True)

#Load logreg model!
logreg=LogisticRegression(multi_class="multinomial",solver="newton-cg",
                                    max_iter=500,penalty='l2',C=1.0,class_weight=None) #C=, class_weights="balanced", "none"


