# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:48:26 2024

@author: Kenny
"""


    
# Imports
from Scripts.util import (load_object_from_file,
                          get_filepath,
                          plot_aggregated_confusion_matrix,
                          write_output_to_csv,
                          calculate_accuracy)

from scipy.stats import ttest_ind

import os
import pandas as pd
import numpy as np
import itertools

def extract_metrics(report,accuracy,model_name):
    macro_avg_precision = report.loc['macro avg', 'precision']
    weighted_avg_precision = report.loc['weighted avg', 'precision']
    macro_avg_recall = report.loc['macro avg', 'recall']
    weighted_avg_recall = report.loc['weighted avg', 'recall']
    macro_avg_f1 = report.loc['macro avg', 'f1-score']
    weighted_avg_f1 = report.loc['weighted avg', 'f1-score']
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Macro Avg Precision': macro_avg_precision,
        'Weighted Avg Precision': weighted_avg_precision,
        'Macro Avg Recall': macro_avg_recall,
        'Weighted Avg Recall': weighted_avg_recall,
        'Macro Avg F1-Score': macro_avg_f1,
        'Weighted Avg F1-Score': weighted_avg_f1
    }
    return metrics



def run_visualization(shading=True):
    pass

#%%Plot Confusion Matrix

# Shading True (Dataset A)
shading = True #PUT INTO FUNCTION LATER!!!

#Define Filename
if shading:
    file_name="Confusion Matrix Dataset A"
else:
    file_name="Confusion Matrix Dataset B"

# Load confusion matrix (from gridsearch)
cm_LR = load_object_from_file("TestTrain_CM_LR.pk1",
                              to_file="LR", shading=shading)

cm_DT = load_object_from_file("TestTrain_CM_DT.pk1",
                              to_file="DT", shading=shading)

cm_RF = load_object_from_file("TestTrain_CM_RF.pk1",
                              to_file="RF", shading=shading)

cm_SVM = load_object_from_file("TestTrain_CM_SVM.pk1",
                               to_file="SVM", shading=shading)

cm_NN = load_object_from_file("TestTrain_CM_NN.pk1",
                              to_file="NN", shading=shading)

plot_aggregated_confusion_matrix(cm_LR, cm_DT, cm_RF, cm_SVM, cm_NN,
                                 title=file_name,to_file="FINAL",
                                 show_plot=False)

# Calculate each single accuracy of all runs (for later)
accuracy_LR = calculate_accuracy(cm_LR)
accuracy_DT = calculate_accuracy(cm_DT)
accuracy_RF = calculate_accuracy(cm_RF)
accuracy_SVM = calculate_accuracy(cm_SVM)
accuracy_NN = calculate_accuracy(cm_NN)

"""
## Shading False (Dataset B)
shading = False

# Load confusion matrix (from gridsearch)
cm_LR = load_object_from_file("TestTrain_CM_LR.pk1",
                              to_file="LR", shading=shading)

cm_DT = load_object_from_file("TestTrain_CM_DT.pk1",
                              to_file="DT", shading=shading)

cm_RF = load_object_from_file("TestTrain_CM_RF.pk1",
                              to_file="RF", shading=shading)

cm_SVM = load_object_from_file("TestTrain_CM_SVM.pk1",
                               to_file="SVM", shading=shading)

cm_NN = load_object_from_file("TestTrain_CM_NN.pk1",
                              to_file="NN", shading=shading)

plot_aggregated_confusion_matrix(cm_LR, cm_DT, cm_RF, cm_SVM, cm_NN,
                                 title=file_name,to_file="FINAL",
                                 show_plot=True)

# Calculate each single accuracy (for later)
accuracy_LR_F = calculate_accuracy(cm_LR)
accuracy_DT_F = calculate_accuracy(cm_DT)
accuracy_RF_F = calculate_accuracy(cm_RF)
accuracy_SVM_F = calculate_accuracy(cm_SVM)
accuracy_NN_F = calculate_accuracy(cm_NN)
"""

#%% Big tables!

#load in reports
report_LR = load_object_from_file("TestTrain_report_LR.pk1",
                                   to_file="LR",shading=shading)

report_DT = load_object_from_file("TestTrain_report_DT.pk1",
                                   to_file="DT",shading=shading)

report_RF = load_object_from_file("TestTrain_report_RF.pk1",
                                   to_file="RF",shading=shading)

report_SVM = load_object_from_file("TestTrain_report_SVM.pk1",
                                   to_file="SVM",shading=shading)

report_NN = load_object_from_file("TestTrain_report_NN.pk1",
                                   to_file="NN",shading=shading)

#extract metrics
metrics_LR=extract_metrics(report_LR,accuracy_LR, "LR")
metrics_DT=extract_metrics(report_DT,accuracy_DT, "DT")
metrics_RF=extract_metrics(report_RF,accuracy_RF, "RF")
metrics_SVM=extract_metrics(report_SVM,accuracy_SVM, "SVM")
metrics_NN=extract_metrics(report_NN, accuracy_NN, "NN")

#Combine data
metrics_data=[metrics_LR,metrics_DT,metrics_RF,metrics_SVM,metrics_NN]

#Create new dataframe
metrics_df = pd.DataFrame(metrics_data)

#Express weighted scores as difference to macro scores
metrics_df["Weighted Avg Precision"]=metrics_df["Weighted Avg Precision"]- metrics_df["Macro Avg Precision"]
metrics_df["Weighted Avg Recall"]=metrics_df["Weighted Avg Recall"]- metrics_df["Macro Avg Recall"]
metrics_df["Weighted Avg F1-Score"]=metrics_df["Weighted Avg F1-Score"]- metrics_df["Macro Avg F1-Score"]

#*100 for percentage values and round to two values after decimal point
metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:] * 100
metrics_df.iloc[:,1:]=metrics_df.iloc[:,1:].round(2)

#Define File-name
if shading:
    file_name ="Metrics Dataset A"    
else:
    file_name ="Metrics Dataset B"

#write to csv
write_output_to_csv(metrics_df,file_name=file_name,to_file="FINAL",shading=None)


#%% Statistical tests

def run_t_tests():
    pass

#read-in data

shading=True
#LR
file_path=get_filepath(model_sd="LR",shading=shading)
file_path+=r"\Test-train-results_LR.csv"
tt_results_LR = pd.read_csv(file_path,sep=";")

#DT
file_path=get_filepath(model_sd="DT",shading=shading)
file_path+=r"\Test-train-results_DT.csv"
tt_results_DT = pd.read_csv(file_path,sep=";")

#RF
file_path=get_filepath(model_sd="RF",shading=shading)
file_path+=r"\Test-train-results_RF.csv"
tt_results_RF = pd.read_csv(file_path,sep=";")

#SVM
file_path=get_filepath(model_sd="SVM",shading=shading)
file_path+=r"\Test-train-results_SVM.csv"
tt_results_SVM = pd.read_csv(file_path,sep=";")

#NN
file_path=get_filepath(model_sd="NN",shading=shading)
file_path+=r"\Test-train-results_NN.csv"
tt_results_NN = pd.read_csv(file_path,sep=";")


#Extract Accuracy values and calculate mean values for Dataset A
c_to_ext="recall_weighted avg"

acc_LR_A=tt_results_LR[c_to_ext].values
mean_LR_A=np.mean(acc_LR_A)

acc_DT_A=tt_results_DT[c_to_ext].values
mean_DT_A=np.mean(acc_DT_A)

acc_RF_A=tt_results_RF[c_to_ext].values
mean_RF_A=np.mean(acc_RF_A)

acc_SVM_A=tt_results_SVM[c_to_ext].values
mean_SVM_A=np.mean(acc_SVM_A)

acc_NN_A=tt_results_NN[c_to_ext].values
mean_NN_A=np.mean(acc_NN_A)


shading = False
file_path=get_filepath(model_sd="LR",shading=shading)
file_path+=r"\Test-train-results_LR.csv"
tt_results_LR = pd.read_csv(file_path,sep=";")

#DT
file_path=get_filepath(model_sd="DT",shading=shading)
file_path+=r"\Test-train-results_DT.csv"
tt_results_DT = pd.read_csv(file_path,sep=";")

#RF
file_path=get_filepath(model_sd="RF",shading=shading)
file_path+=r"\Test-train-results_RF.csv"
tt_results_RF = pd.read_csv(file_path,sep=";")

#SVM
file_path=get_filepath(model_sd="SVM",shading=shading)
file_path+=r"\Test-train-results_SVM.csv"
tt_results_SVM = pd.read_csv(file_path,sep=";")

#NN
file_path=get_filepath(model_sd="NN",shading=shading)
file_path+=r"\Test-train-results_NN.csv"
tt_results_NN = pd.read_csv(file_path,sep=";")

#Extract accuracy values for dataset B
acc_LR_B=tt_results_LR[c_to_ext].values
mean_LR_B=np.mean(acc_LR_B)

acc_DT_B=tt_results_DT[c_to_ext].values
mean_DT_B=np.mean(acc_DT_B)

acc_RF_B=tt_results_RF[c_to_ext].values
mean_RF_B=np.mean(acc_RF_B)

acc_SVM_B=tt_results_SVM[c_to_ext].values
mean_SVM_B=np.mean(acc_SVM_B)

acc_NN_B=tt_results_NN[c_to_ext].values
mean_NN_B=np.mean(acc_NN_B)


#Generating possible combinations

#Dataset A
acc_datasetA=[acc_LR_A, 
              acc_DT_A,
              acc_RF_A,
              acc_SVM_A,
              acc_NN_A]

acc_datasetA_names=["acc_LR_A", 
                    "acc_DT_A",
                    "acc_RF_A",
                    "acc_SVM_A",
                    "acc_NN_A"]

combinations_A = list(itertools.combinations(zip(acc_datasetA, acc_datasetA_names), 2))

#Dataset B
acc_datasetB=[acc_LR_B, 
              acc_DT_B,
              acc_RF_B,
              acc_SVM_B,
              acc_NN_B]

acc_datasetB_names=["acc_LR_B", 
                    "acc_DT_B",
                    "acc_RF_B",
                    "acc_SVM_B",
                    "acc_NN_B"]

combinations_B = list(itertools.combinations(zip(acc_datasetB,acc_datasetB_names), 2))

print(combinations_A[0][0][1])
print(combinations_A[0][1][1])

"""
combinations is a nested list where each item contains a tuple and where each 
value of the tuple is again a tuple
combinations [0-9][0/1][0/1] first indicates list index second indicates if first
or second sample is accessed, third indicates if name [1] or actual np.array [0]
is accessed
"""
# for index,item in enumerate(combinations_A):
#     print(f'index = {index}\nitem0 = {item[0][1]}\nitem1={item[1][1]}')
    
#Create a new dataframe with the right values 
#First create empty list where DF is constructed later!
index_list = []
tuple_list = []
sample_A_list = []
sample_B_list = []

# Iterate over combinationsA to populate lists
for list_index, item in enumerate(combinations_A):
    index_list.append(list_index)
    tuple_list.append((item[0][1], item[1][1]))  # Tuple with NAME of combination
    sample_A_list.append(item[0][0])  # NP array of first Tuple (sample A)
    sample_B_list.append(item[1][0])  # NP array of second Tuple (sample B)
    
#Create a dictonary with lists to construct the df

dictonary_A = {
    'list_index': index_list,
    'combination': tuple_list,
    'sample_A': sample_A_list,
    'sample_B': sample_B_list
}


df_data_A=pd.DataFrame(dictonary_A)


#Perform one t-test

t_stat, p_value = ttest_ind(sample_A_list[0], sample_B_list[0])
print(f"Independent t-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")

# Interpretation
if p_value < 0.05:
    print("There is a significant difference between the accuracy scores of the two models.")
else:
    print("There is no significant difference between the accuracy scores of the two models.")
