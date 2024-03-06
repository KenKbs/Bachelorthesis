# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:21:50 2024

@author: Kenny
"""

#%% Imports 
#Main package imports done in util.py

#Internal imports
from Scripts.util import (
    get_data,
    filter_data,
    generate_table,
    train_test_split_data,
    perform_grid_search,
    get_performance_metrics,
    get_confusion_matrix,
    plot_confusion_matrix,
    write_output_to_csv,
    save_object_to_file
)

#SK-learn model imports
from sklearn.linear_model import LogisticRegression



#%% Read in and filter data

#Read-in Data
raw_data=get_data()

#Filter out low irrediance values
data=filter_data(raw_data,filter_value=100)

#Print fault distribution before and after fitering
generate_table(raw_data,data,"raw","filtered") 


#%%Grid search preparations

# Split data w. own fuinction, scaling = True
x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                         test_size=0.2,scaling=True)

#Define Model to tune (Logistic Regression)
logreg=LogisticRegression(multi_class="multinomial",solver="newton-cg",
                                    max_iter=500,penalty='l2',C=1.0,class_weight=None) #C=, class_weights="balanced", "none"

#Set Parameter to tune

# penalization parameter 
penalization_param=[0.0001,0.001,0.01,0.03,0.05,0.07,0.09,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                    1,2,3,4,5,6,7,8,9,
                    10,20,30,40,50,60,70,80,90,
                    100,200,300,400,500,600,700,800,900,
                    1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,
                    3000,3250,3500,3750,4000,4250,4500,4750,
                    5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,
                    12500,15000,17500,20000,22500,25000,27500,30000,
                    40000,50000,60000,70000,80000,90000,100000,
                    200000,300000,400000,500000,600000,700000,800000,900000,
                    1000000]

#Give over parameters to parameter grid to search for
param_grid={'C':penalization_param}

#TESTING
param_grid_t={'C':[1,1000]} #REMOVE LATER AND CHANGE FUNCTION CALL!

#%%Perform Grid_search
best_model,cv_results=perform_grid_search(x_train,y_train,logreg,param_grid_t)


#%%Performance evaluation

#Predictions on test-set
y_pred_best=best_model.predict(x_test)

#Get Accuracy Score
accuracy=get_performance_metrics(y_test, y_pred_best,only_accuracy=True)

#Get f1,recall,precision etc.
report=get_performance_metrics(y_test, y_pred_best) #report=df

#Get confusion Matrix
cm=get_confusion_matrix(y_test, y_pred_best,normalize=False)

#plot Confusion Matrix and save to file
plot_confusion_matrix(cm,to_file="LR",normalize=True,shading=True)

#save best model to file
save_object_to_file(best_model,file_name="Best_Model",
                    to_file="LR",shading=True)

#save confusion matrix to file:
save_object_to_file(cm,file_name="Grid-search_CM",
                    to_file="LR",shading=True)

#save report (f1_score etc.) to file:
save_object_to_file(report,file_name="Grid-search_report",
                    to_file="LR",shading=True)

#Save Grid-search results to csv_file
write_output_to_csv(cv_results,output2=report.round(4), #take rounded numbers of report for better overview
                    file_name="Grid-search-results",
                    to_file="LR",shading=True)



