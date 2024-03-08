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
    get_filepath,
    train_test_split_data,
    load_object_from_file,
    get_performance_metrics,
    get_confusion_matrix,
    plot_confusion_matrix,
    write_output_to_csv,
    save_object_to_file)

from sklearn.linear_model import LogisticRegression

import numpy as np

shading = True #put later into function
num_iterations=2 #times how often train/test split also put later into function
run=1 #Counter for numbers of runs

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

#%%Initialize csv-file
#Manipulate report_all to prepare for writing to csv:
single_row=report_all.values.flatten()
single_row=np.insert(single_row,0,run)#Insert current run as first value

#Create custom row labels based on index and column names first four = iterable index last two = index name of df
row_labels = [f'{col}_{ix}' if ix <= 4 else f'{col}_{idx}' for ix, idx in enumerate(report_all.index) for col in report_all.columns]
row_labels.insert(0,'run_counter') #Insert run column

#Intialize csv_file with first row of report_all (results from gridsearch)
raw_file_path=get_filepath(model_sd="LR",shading=shading)
file_path=raw_file_path+r'\Test-train-results_LR.csv'

#open file outside loop once
with open(file_path,'w') as file:
    file.write(';'.join(row_labels)+'\n')
    file.write(';'.join(map(str,single_row))+'\n')


#%%Rpeaditly train_test data
    for i in range (num_iterations):
        #Change run counter
        run=i+2 #because first "run" = results of Gridsearch
        
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
        
        #Write results of this run to csv
        single_row=report_tt.values.flatten()
        single_row=np.insert(single_row,0,run) #insert run in first position
        file.write(';'.join(map(str,single_row))+'\n')
    ##END LOOP 
    #Append AGGREGATED REPORT to CSV to the end
    # single_row=report_all.values.flatten()
    # single_row=np.insert(single_row,0,'9999')
    # file.write('\n\n\n')
    # file.write('Aggregated:\n')
    # file.write(';'.join(map(str,single_row)))
    
##CLOSE FILE
#Write aggregated report to seperate csv_file
params=logreg.get_params()
write_output_to_csv(report_all,cm_all,params,file_name="Test-train-aggregated-results_LR",
                    to_file="LR",shading=shading)

print(f'All data has been sucessfully written to files \nSee Filepath:\n {raw_file_path}')
    
#%% Save aggregated Results to file / pdf

#plot Confusion Matrix and save to pdf
plot_confusion_matrix(cm_all,to_file="LR",show_plot=True,normalize=True,
                      shading=shading,
                      title="TestTrain ConfusionMatrix LR")
     
#save confusion matrix to file:
save_object_to_file(cm_all,file_name="TestTrain_CM_LR",
                    to_file="LR",shading=shading)

#save (aggregated) report (f1_score etc.) to file:
save_object_to_file(report_all,file_name="TestTrain_report_LR",
                    to_file="LR",shading=shading)

### We need to add a way to track "manual" scores


