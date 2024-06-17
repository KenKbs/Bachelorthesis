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
                          calculate_accuracy,
                          save_object_to_file,
                          plot_training_time)

import pandas as pd
import numpy as np
import itertools
from scipy.stats import ttest_ind


#Internal function
def _extract_metrics(report,accuracy,model_name):
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


#%%Plot Confusion Matrix

def plot_cmANDcreate_metrics(shading=True):   
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
    
    print(f"{file_name} sucessfully plotted and saved to directory Results/FINAL \n")
    
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
    metrics_LR=_extract_metrics(report_LR,accuracy_LR, "LR")
    metrics_DT=_extract_metrics(report_DT,accuracy_DT, "DT")
    metrics_RF=_extract_metrics(report_RF,accuracy_RF, "RF")
    metrics_SVM=_extract_metrics(report_SVM,accuracy_SVM, "SVM")
    metrics_NN=_extract_metrics(report_NN, accuracy_NN, "NN")
    
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
    
    print(f"{file_name} successfully created and saved to Results/FINAL \n")

#%% Statistical tests

def run_t_testsANDplot_training_time():
    
    #read-in data for dataset A (shading = True)    
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
    
    
    #Extract Accuracy values for Dataset A
    c_to_ext="accuracy"
    
    acc_LR_A=tt_results_LR[c_to_ext].values
    acc_DT_A=tt_results_DT[c_to_ext].values
    acc_RF_A=tt_results_RF[c_to_ext].values
    acc_SVM_A=tt_results_SVM[c_to_ext].values
    acc_NN_A=tt_results_NN[c_to_ext].values
    
    
    #Extract training times
    time_LR_A=tt_results_LR["train or test time in seconds"].values
    time_DT_A=tt_results_DT["train or test time in seconds"].values
    time_RF_A=tt_results_RF["train or test time in seconds"].values
    time_SVM_A=tt_results_SVM["train or test time in seconds"].values
    time_NN_A=tt_results_NN["train or test time in seconds"].values
    
    
    
    #read in data for Dataset B (shading = False)
    shading = False
    
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
    
    #Extract accuracy values (np arrays) for dataset B
    acc_LR_B=tt_results_LR[c_to_ext].values
    acc_DT_B=tt_results_DT[c_to_ext].values
    acc_RF_B=tt_results_RF[c_to_ext].values
    acc_SVM_B=tt_results_SVM[c_to_ext].values
    acc_NN_B=tt_results_NN[c_to_ext].values
    
    #Extract training times for dataset B
    time_LR_B=tt_results_LR["train or test time in seconds"].values
    time_DT_B=tt_results_DT["train or test time in seconds"].values
    time_RF_B=tt_results_RF["train or test time in seconds"].values
    time_SVM_B=tt_results_SVM["train or test time in seconds"].values
    time_NN_B=tt_results_NN["train or test time in seconds"].values
    
    
    #Generating possible combinations for t-tests
    
    #put Dataset A accuracy arrays into a list
    acc_datasetA=[acc_LR_A, 
                  acc_DT_A,
                  acc_RF_A,
                  acc_SVM_A,
                  acc_NN_A]
    
    #create a list with names
    acc_datasetA_names=["acc_LR_A", 
                        "acc_DT_A",
                        "acc_RF_A",
                        "acc_SVM_A",
                        "acc_NN_A"]
    
    #create all combinations of models for Dataset A
    combinations_A = list(itertools.combinations(zip(acc_datasetA, acc_datasetA_names), 2))
    
    #Same for Dataset B
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
    
    #Create all combinations of models for Dataset B
    combinations_B = list(itertools.combinations(zip(acc_datasetB,acc_datasetB_names), 2))
    
    
    #Create combinations with same model (LR_A,LR_B), same structure as combinations_A,_B List! (therefore double zipped)
    combinations_SAME=list(zip(zip(acc_datasetA,acc_datasetA_names),zip(acc_datasetB,acc_datasetB_names)))
    
    
    """
    combinations is a nested list where each item contains a tuple and where each 
    value of the tuple is again a tuple
    combinations [0-9][0/1][0/1] first indicates list index second indicates if first
    or second sample is accessed, third indicates if name [1] or actual np.array [0]
    is accessed
    print(combinations_A[0][0][1])
    print(combinations_A[0][1][1])
    """
        
    
    
    #%%COMBINATIONS_A
    #Create a new dataframes which contain necessary information (t-test values etc.)
    
    #First create empty list from which DF is constructed later!
    index_list = [] #store index of list
    tuple_list = [] #store names of combinations
    avg_accuracy_tuple_list=[] #stores tuples with mean accuracy of combination
    sample_A_list = [] #store first np array 
    sample_B_list = [] #store second np array
    t_statistics_list =[] #store t-statistics
    p_value_list =[] #store p-value
    significant_list=[] #yes/no
    
    # Iterate over combinationsA to populate lists
    for list_index, item in enumerate(combinations_A):
        index_list.append(list_index)
        tuple_list.append((item[0][1], item[1][1]))  # Tuple with NAME of combination
        avg_accuracy_tuple_list.append((np.mean(item[0][0]),np.mean(item[1][0]))) #tuple (avg. accuracy sample A, avg. accuracy sample B)
        sample_A_list.append(item[0][0])  # NP array of first Tuple (sample A)
        sample_B_list.append(item[1][0])  # NP array of second Tuple (sample B)
        
        
    #Perform t-tests on sample_A_list and sample_B_list to populate t_statistics and p_value
    for i in range(len(sample_A_list)):
        t_stat, p_value = ttest_ind(sample_A_list[i], sample_B_list[i]) #perform t-test over A and B
        t_statistics_list.append(t_stat)
        p_value_list.append(p_value)
        print(f"Independent t-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
    
        # Interpretation a=5%
        if p_value < 0.05:
            significant_list.append("yes")
        else:
            significant_list.append("no")
    
    #Create a dictonary with lists to construct the df
    dictonary_A = {
        'list_index': index_list,
        'combination': tuple_list,
        'avg_accuracy':avg_accuracy_tuple_list,
        'sample_A': sample_A_list,
        'sample_B': sample_B_list,
        't_statistics':t_statistics_list,
        'p_value':p_value_list,
        'significant 5%':significant_list
    }
    
    #Construct DF
    df_ttest_A=pd.DataFrame(dictonary_A)
    
    #Pickle DF
    save_object_to_file(df_ttest_A, file_name="ttest-A",to_file="FINAL",shading=None)
    
    #Drop sample_A, sample_B columns for saving to csv (else it's too long)
    df_ttest_A=df_ttest_A.drop(columns=["sample_A","sample_B"])
    
    #Write to csv
    write_output_to_csv(df_ttest_A,file_name="ttest-A",to_file="FINAL",shading=None)
    
    
    #%%COMBINATIONS B
    
    #Again, Create empty list from which DF is constructed later!
    index_list = [] #store index of list
    tuple_list = [] #store names of combinations
    avg_accuracy_tuple_list=[] #stores tuples with mean accuracy of combination
    sample_A_list = [] #store first np array 
    sample_B_list = [] #store second np array
    t_statistics_list =[] #store t-statistics
    p_value_list =[] #store p-value
    significant_list=[] #yes/no
    
    # Iterate over combinationsB to populate lists
    for list_index, item in enumerate(combinations_B):
        index_list.append(list_index)
        tuple_list.append((item[0][1], item[1][1]))  # Tuple with NAME of combination
        avg_accuracy_tuple_list.append((np.mean(item[0][0]),np.mean(item[1][0]))) #tuple (avg. accuracy sample A, avg. accuracy sample B)
        sample_A_list.append(item[0][0])  # NP array of first Tuple (sample A)
        sample_B_list.append(item[1][0])  # NP array of second Tuple (sample B)
        
        
    #Perform t-tests on sample_A_list and sample_B_list to populate t_statistics and p_value
    for i in range(len(sample_A_list)):
        t_stat, p_value = ttest_ind(sample_A_list[i], sample_B_list[i]) #perform t-test over A and B
        t_statistics_list.append(t_stat)
        p_value_list.append(p_value)
        print(f"Independent t-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
    
        # Interpretation a=5%
        if p_value < 0.05:
            significant_list.append("yes")
        else:
            significant_list.append("no")
    
    #Create a dictonary with lists to construct the df
    dictonary_B = {
        'list_index': index_list,
        'combination': tuple_list,
        'avg_accuracy':avg_accuracy_tuple_list,
        'sample_A': sample_A_list,
        'sample_B': sample_B_list,
        't_statistics':t_statistics_list,
        'p_value':p_value_list,
        'significant 5%':significant_list
    }
    
    #Construct DF
    df_ttest_B=pd.DataFrame(dictonary_B)
    
    #Pickle DF
    save_object_to_file(df_ttest_B, file_name="ttest-B",to_file="FINAL",shading=None)
    
    #Drop sample_A, sample_B columns for saving to csv (else it's too long)
    df_ttest_B=df_ttest_B.drop(columns=["sample_A","sample_B"])
    
    #Write to csv
    write_output_to_csv(df_ttest_B,file_name="ttest-B",to_file="FINAL",shading=None)
    
    
    #%%COMBINATIONS SAME
    
    #Create empty lists again
    index_list = [] #store index of list
    tuple_list = [] #store names of combinations
    avg_accuracy_tuple_list=[] #stores tuples with mean accuracy of combination
    sample_A_list = [] #store first np array 
    sample_B_list = [] #store second np array
    t_statistics_list =[] #store t-statistics
    p_value_list =[] #store p-value
    significant_list=[] #yes/no
    
    # Iterate over combinations_SAME to populate lists
    for list_index, item in enumerate(combinations_SAME):
        index_list.append(list_index)
        tuple_list.append((item[0][1], item[1][1]))  # Tuple with NAME of combination
        avg_accuracy_tuple_list.append((np.mean(item[0][0]),np.mean(item[1][0]))) #tuple (avg. accuracy sample A, avg. accuracy sample B)
        sample_A_list.append(item[0][0])  # NP array of first Tuple (sample A)
        sample_B_list.append(item[1][0])  # NP array of second Tuple (sample B)
        
        
    #Perform t-tests on sample_A_list and sample_B_list to populate t_statistics and p_value
    for i in range(len(sample_A_list)):
        t_stat, p_value = ttest_ind(sample_A_list[i], sample_B_list[i]) #perform t-test over A and B
        t_statistics_list.append(t_stat)
        p_value_list.append(p_value)
        print(f"Independent t-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
    
        # Interpretation a=5%
        if p_value < 0.05:
            significant_list.append("yes")
        else:
            significant_list.append("no")
    
    #Create a dictonary with lists to construct the df
    dictonary_SAME = {
        'list_index': index_list,
        'combination': tuple_list,
        'avg_accuracy':avg_accuracy_tuple_list,
        'sample_A': sample_A_list,
        'sample_B': sample_B_list,
        't_statistics':t_statistics_list,
        'p_value':p_value_list,
        'significant 5%':significant_list
    }
    
    #Construct DF
    df_ttest_SAME=pd.DataFrame(dictonary_SAME)
    
    #Pickle DF
    save_object_to_file(df_ttest_SAME, file_name="ttest-SAME",to_file="FINAL",shading=None)
    
    #Drop sample_A, sample_B columns for saving to csv (else it's too long)
    df_ttest_SAME=df_ttest_SAME.drop(columns=["sample_A","sample_B"])
    
    #Write to csv
    write_output_to_csv(df_ttest_SAME,file_name="ttest-SAME",to_file="FINAL",shading=None)
    
    """
    Not really elegant coding, would make it more pretty if more time...
    """
    
    print("All t-tests sucessfully conducted, saved csv to Results/FINAL \n")
    
    
    #%% Plot training time
    
    #calculate mean training times
    #dataset A
    mean_time_LR_A=np.mean(time_LR_A)
    mean_time_DT_A=np.mean(time_DT_A)
    mean_time_RF_A=np.mean(time_RF_A)
    mean_time_SVM_A=np.mean(time_SVM_A)
    mean_time_NN_A=np.mean(time_NN_A)
    
    #dataset B
    mean_time_LR_B=np.mean(time_LR_B)
    mean_time_DT_B=np.mean(time_DT_B)
    mean_time_RF_B=np.mean(time_RF_B)
    mean_time_SVM_B=np.mean(time_SVM_B)
    mean_time_NN_B=np.mean(time_NN_B)
    
    
    #Create lists with names and mean training time
    model_names=["LR","DT","RF","SVM","NN"]
    
    mean_time_A_list=[mean_time_LR_A,mean_time_DT_A, 
                      mean_time_RF_A, mean_time_SVM_A, 
                      mean_time_NN_A]
    
    mean_time_B_list=[mean_time_LR_B,mean_time_DT_B, 
                      mean_time_RF_B, mean_time_SVM_B, 
                      mean_time_NN_B]
    
    #Plot training time
    plot_training_time(mean_time_A_list, mean_time_B_list, model_names,
                       show_plot=False,to_file="FINAL")
    
   
    print("Training time sucessfully plotted and saved to Results/FINAL \n")
    


#%% MANUELLE ÜBERPRÜFUNG --> stimmt alles :))
"""
import math

A=acc_DT_A
B=acc_NN_A

sigmaA=np.var(A)
sigmaB=np.var(B)

meanA=np.mean(A)
meanB=np.mean(B)


sigma=(1/(100+100-2))*((99*sigmaA)+99*sigmaB)

T_value=(meanA-meanB)/math.sqrt(sigma*(1/100+1/100))

print(f"sigma:{sigma} and T-value:{T_value}")

print(f"meanA: {meanA}   meanB: {meanB}")

#Highly significant because of extremly small variances!
"""

