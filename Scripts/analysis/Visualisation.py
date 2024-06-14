# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:48:26 2024

@author: Kenny
"""


    
# Imports
from Scripts.util import (load_object_from_file,
                          plot_aggregated_confusion_matrix)

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

def calculate_accuracy(cm):
    cm=cm.values
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    return accuracy

#%%Plot Confusion Matrix

# Shading True (Dataset A)
shading = True

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
                                 title="Confusion Matrix Dataset A",to_file="FINAL",
                                 show_plot=True)

# Calculate each single accuracy (for later)
accuracy_LR_T = calculate_accuracy(cm_LR)
accuracy_DT_T = calculate_accuracy(cm_DT)
accuracy_RF_T = calculate_accuracy(cm_RF)
accuracy_SVM_T = calculate_accuracy(cm_SVM)
accuracy_NN_T = calculate_accuracy(cm_NN)


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
                                 title="Confusion Matrix Dataset B",to_file="FINAL",
                                 show_plot=True)

# Calculate each single accuracy (for later)
accuracy_LR_F = calculate_accuracy(cm_LR)
accuracy_DT_F = calculate_accuracy(cm_DT)
accuracy_RF_F = calculate_accuracy(cm_RF)
accuracy_SVM_F = calculate_accuracy(cm_SVM)
accuracy_NN_F = calculate_accuracy(cm_NN)


#%% Big tables!

# shading True (Dataset A)
shading = True

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
metrics_LR=extract_metrics(report_LR,accuracy_LR_T, "LR")
metrics_DT=extract_metrics(report_DT,accuracy_DT_T, "DT")
metrics_RF=extract_metrics(report_RF,accuracy_RF_T, "RF")
metrics_SVM=extract_metrics(report_SVM,accuracy_SVM_T, "SVM")
metrics_NN=extract_metrics(report_NN, accuracy_NN_T, "NN")

#Combine data
metrics_data=[metrics_LR,metrics_DT,metrics_RF,metrics_SVM,metrics_NN]

#Create new dataframe
metrics_df_T = pd.DataFrame(metrics_data)

#Express weighted scores as difference to macro scores
metrics_df_T["Weighted Avg Precision"]=metrics_df_T["Weighted Avg Precision"]- metrics_df_T["Macro Avg Precision"]
metrics_df_T["Weighted Avg Recall"]=metrics_df_T["Weighted Avg Recall"]- metrics_df_T["Macro Avg Recall"]
metrics_df_T["Weighted Avg F1-Score"]=metrics_df_T["Weighted Avg F1-Score"]- metrics_df_T["Macro Avg F1-Score"]

#*100 for percentage values and round to two values after decimal point
metrics_df_T.iloc[:, 1:] = metrics_df_T.iloc[:, 1:] * 100
metrics_df_T.iloc[:,1:]=metrics_df_T.iloc[:,1:].round(2)

#shading False (Dataset B)


