# -*- coding: utf-8 -*-
"""
Created on Wed Apr. 04 16:24:46 2024

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
def run_RF_traintest(shading=True,num_iterations=100):
    """
    Repeaditly trains and tests best RF model and saves
    results to "RF" directory.
    
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
    
    #Counter for numbers of runs = n
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
                
    #Load DT model with pickle
    random_forest=load_object_from_file("Best_Model_RF.pk1",
                                 to_file="RF",shading=shading)
    
    #load report (from Gridsearch) as "starting" value
    report_all=load_object_from_file("Grid-search_report_RF.pk1",
                                         to_file="RF",shading=shading)
    
    #load confusion matrix (from gridserach)
    cm_all=load_object_from_file("Grid-search_CM_RF.pk1",
                                         to_file="RF",shading=shading)
    
    
    #%%Initialize csv-file
    #Manipulate report_all to prepare for writing to csv:
        
    #Convert report_all to single row
    single_row=convert_to_srow(report_all,run)
    
    #Create custom row labels based on index and column names first four = iterable index last two = index name of df
    row_labels=convert_to_srow(df=report_all,insert_value='run_counter',
                               extract_labels=True)
    
    #Get file_path and attach filename
    parent_file_path=get_filepath(model_sd="RF",shading=shading)
    file_path=parent_file_path+r'\Test-train-results_RF.csv'
    
    #Choose columns to average for the report
    ctavg=['precision','recall','f1-score'] 
    
    #Intialize csv_file with first row of report_all (results from gridsearch)
    #Open file outside loop once and close after loop
    with open(file_path,'w') as file:
        file.write(';'.join(row_labels)+'\n')
        file.write(';'.join(map(str,single_row))+'\n')
    
        #Print Progress (first run was = Gridsearch, therefore first run completed before loop)
        print(f'Run {run} successfully completed \n')
    
    #%%Rpeaditly train_test data
        for i in range (num_iterations):
            #Change run counter
            run=i+2 #because first "run" = results of Gridsearch also equal to n
            
            # Split data w. own fuinction, scaling = True
            x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                                     test_size=0.2,
                                                                     scaling=False)  #no z-transformation anymore  
            #Refit to new data
            random_forest.fit(x_train,y_train)
                
            #Evaluation and Result manipulation    
            #Predict classes
            y_pred=random_forest.predict(x_test)
            
            #Get f1,recall,precision etc.as DF
            report_tt=get_performance_metrics(y_test, y_pred)
            
            #Get confusion Matrix as DF
            cm_tt=get_confusion_matrix(y_test, y_pred,normalize=False)
            
            #convert tt_Results to single row
            single_row=convert_to_srow(report_tt,run)
            
            #Write results of this run to csv
            file.write(';'.join(map(str,single_row))+'\n')
            
            #Combine Dataframes for aggregated results
            """
            u_n=u_n-1+(x_n-u_n-1)/n
            """
            #report - incremental average approach due to memory for f1-score, precision and recall
            report_all[ctavg]=report_all[ctavg]+((report_tt[ctavg]-report_all[ctavg])/run)
            
            #report - add up support column (number of cases)
            report_all['support']=report_all['support']+report_tt['support']

            
            #Add up Confusion Matrix
            cm_all=cm_all+cm_tt
            
            #print progress
            print(f' Run {run} successfully completed \n')
            
        ##END LOOP 
    ##CLOSE FILE
    
    #%% Save aggregated Results to file / pdf
    
    #Write aggregated report, cm etc. to seperate csv_file
    params=random_forest.get_params() #get params of model
    write_output_to_csv(report_all.round(4),cm_all,params,file_name="Test-train-aggregated-results_RF",
                        to_file="RF",shading=shading)
      
    #plot Confusion Matrix and save to pdf
    plot_confusion_matrix(cm_all,to_file="RF",show_plot=False,normalize=True,
                          shading=shading,
                          title=f"TestTrain ConfusionMatrix RF shading {shading}")
         
    #save (absolute) confusion matrix to file:
    save_object_to_file(cm_all,file_name="TestTrain_CM",
                        to_file="RF",shading=shading)
    
    #save (aggregated) report (f1_score etc.) to file:
    save_object_to_file(report_all,file_name="TestTrain_report",
                        to_file="RF",shading=shading)
    
    #Print result
    print(f'All data has been sucessfully written to files \nSee Filepath:\n {parent_file_path}')
