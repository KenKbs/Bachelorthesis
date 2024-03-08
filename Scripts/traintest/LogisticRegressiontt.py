# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:31:16 2024

@author: Kenny
"""
#%% Imports
from Scripts.util import (
    get_data,
    filter_data,
    generate_table,
    train_test_split_data,
    load_object_from_file,
    get_performance_metrics,
    get_confusion_matrix,
    plot_confusion_matrix,
    write_output_to_csv)

from sklearn.linear_model import LogisticRegression


shading = True #put later into function
num_iterations=20 #times how often train/test split also put later into function

#%% data and model preperations

#Read in Data
raw_data=get_data()

#Filter data
data=filter_data(raw_data,filter_value=100,shading=shading)

#Print out fault distribution before and after filtering
    #Include shading cases
if shading:
    generate_table(raw_data,data,"Raw","Filtered")
#Exclude shading casees
else:
    generate_table(raw_data,data,"Raw","Shad. excl")
            
#Load logreg model with pickle
logreg=load_object_from_file("Best_Model_LR.pk1",
                             to_file="LR",shading=shading)

#load report (from Gridsearch) as "starting" value
report_all=load_object_from_file("Grid-search_report_LR.pk1",
                                     to_file="LR",shading=shading)

#load confusion matrix (from gridserach)
cm_all=load_object_from_file("Grid-search_CM_LR.pk1",
                                     to_file="LR",shading=shading)


#%%Rpeaditly train_test data

for i in range (num_iterations):

    # Split data w. own fuinction, scaling = True
    x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                             test_size=0.2,scaling=True)    
    #Refit to new data
    logreg.fit(x_train,y_train)
        
    #Evaluation and Result manipulation    
    #Predict classes
    y_pred=logreg.predict(x_test)
    
    #Get f1,recall,precision etc.as DF
    report_tt=get_performance_metrics(y_test, y_pred)
    
    #Get confusion Matrix as DF
    cm_tt=get_confusion_matrix(y_test, y_pred,normalize=False)
    
    #Combine Dataframes 
    #report - incremental average approach due to memory
    report_all=(report_all+report_tt)/2
    
    #Add up Confusion Matrix
    cm_all=cm_all+cm_tt


##END LOOP 
#%% Save Results to file / csv 

#plot Confusion Matrix and save to file
"""
plot_confusion_matrix(cm_all,to_file="LR",show_plot=False,normalize=True,
                      shading=shading,
                      title="TestTrain ConfusionMatrix LR")
     


#save confusion matrix to file:
save_object_to_file(cm_all,file_name="TestTrain_CM",
                    to_file="LR",shading=shading)

#save report (f1_score etc.) to file:
save_object_to_file(report_all,file_name="TestTrain_report",
                    to_file="LR",shading=shading)


"""


### What do we want to print out?
### We need to add a way to track "manual" scores
### --> we need a function, that "dismantels" the dataframe and writes it to a csv table!!!

