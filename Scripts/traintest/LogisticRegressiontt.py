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
    save_object_to_file,
    convert_to_srow)

#%% main function
def run_LR_traintest(shading=True,num_iterations=100):
    """
    Repeaditly trains and tests LR model and saves
    results to "LR" directory.
    
    Parameters
    ----------
    shading : Bool, optional
        with Shading / wo Shading. The default is True.
    num_iterations : Int, optional
        How often train-test is performed and evaluated.
        The default is 100.
        note: the first "run" is always the result from the gridsearch
        so if value = 100 --> 99 runs from repeaditly train/test, one from GS

    Returns
    -------
    None.

    """
    #Initialize starting values
    num_iterations -=1 #For conviniance reasons, results of gridsearch = first "run"    
    
    #Counter for numbers of runs
    run=1 #gets increased with first loop
    
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
        
    #Convert report_all to single row
    single_row=convert_to_srow(report_all,run)
    
    #Create custom row labels based on index and column names first four = iterable index last two = index name of df
    row_labels=convert_to_srow(report_all,'run_counter',extract_labels=True)
    
    #Get file_path
    parent_file_path=get_filepath(model_sd="LR",shading=shading)
    file_path=parent_file_path+r'\Test-train-results_LR.csv'
    
    #Intialize csv_file with first row of report_all (results from gridsearch)
    #Open file outside loop once and close after loop
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
            
            #convert tt_Results to single row
            single_row=convert_to_srow(report_tt,run)
            
            #Write results of this run to csv
            file.write(';'.join(map(str,single_row))+'\n')
            
            #Combine Dataframes for aggregated results
            #report - incremental average approach due to memory
            report_all=(report_all+report_tt)/2
            
            #Add up Confusion Matrix
            cm_all=cm_all+cm_tt
            
        ##END LOOP 
    ##CLOSE FILE
    
    #%% Save aggregated Results to file / pdf
    
    #Write aggregated report, cm etc. to seperate csv_file
    params=logreg.get_params() #get params of model
    write_output_to_csv(report_all.round(4),cm_all,params,file_name="Test-train-aggregated-results_LR",
                        to_file="LR",shading=shading)
      
    #plot Confusion Matrix and save to pdf, also show plot!
    plot_confusion_matrix(cm_all,to_file="LR",show_plot=True,normalize=True,
                          shading=shading,
                          title=f"TestTrain ConfusionMatrix LR shading {shading}")
         
    #save (absolute) confusion matrix to file:
    save_object_to_file(cm_all,file_name="TestTrain_CM_LR",
                        to_file="LR",shading=shading)
    
    #save (aggregated) report (f1_score etc.) to file:
    save_object_to_file(report_all,file_name="TestTrain_report_LR",
                        to_file="LR",shading=shading)
    
    #Print result
    print(f'All data has been sucessfully written to files \nSee Filepath:\n {parent_file_path}')
    