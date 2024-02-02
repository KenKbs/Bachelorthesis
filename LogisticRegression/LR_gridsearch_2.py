# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:26:34 2024

@author: Kenny
"""

#%% Imports:

import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import make_scorer,accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression

from LR_version_2 import get_data
from LR_version_2 import filter_data
from LR_version_2 import train_test_split_data
from LR_version_2 import evaluate_model_performance


#%% functions

def generate_combinations_dict(lists_to_generate=10):
    """
    Generate combinations of weights and convert them to dictionaries.
    Can be passed to Grid-search

    Parameters
    ----------
    lists_to_generate : int, optional
        The number of random weight sets to generate and include in the combinations. 
        The default is 10.

    Returns
    -------
    combinations : list of dict
        A list containing dictionaries representing combinations of weights.
        Each dictionary has keys representing the faults (0-4) and values
        representing the corresponding weight in the permutation.


    """
    #Imports 
    from itertools import permutations
    import random
    
    # Predefined weight sets
    weight_sets = [
        [2,1,1,1,1],
        [2,2,1,1,1],
        [2,2,2,1,1]
        # [1, 2, 3, 4, 5],
        # [100,10,10,10,10],
        # [100,100,10,10,10],    
    ]
    
    # Generate and append random weight sets to weight_sets
    for _ in range(lists_to_generate):
        random_list=[random.uniform(1,10) for _ in range(5)] 
        weight_sets.append(random_list)
        
    
    combinations = []

    #Generate permuatations and convert them to dictonaries
    for weights in weight_sets:
        permutations_list = list(permutations(weights))
        for perm_tuple in permutations_list:
            combination_dict = {i: perm_tuple[i] for i in range(len(perm_tuple))}
            combinations.append(combination_dict)

    return combinations


def extract_class_weights (dataframe):
    """
    Takes a dataframe and extracts the class weights and writes
    them to seperate columns in absolute and normalized values

    Parameters
    ----------
    dataframe : DataFrame
        Takes the results dataframe of grid-search

    Returns
    -------
    dataframe : DataFrame
    appended each class weight to a seperate column

    """
    #iterate over entrys of column param_class_weight
    for index,entry in dataframe['param_class_weight'].items():
        
        #case =  String ('balanced')  
        if isinstance(entry, str): 
            #Balanced strings, add to absolute columns         
            dataframe.loc[index,'0']=entry
            dataframe.loc[index,'1']=entry
            dataframe.loc[index,'2']=entry
            dataframe.loc[index,'3']=entry
            dataframe.loc[index,'4']=entry
            
            #Balanced strings, add to normalized columns
            dataframe.loc[index,'0 normaliz']=entry
            dataframe.loc[index,'1 normaliz']=entry
            dataframe.loc[index,'2 normaliz']=entry
            dataframe.loc[index,'3 normaliz']=entry
            dataframe.loc[index,'4 normaliz']=entry
            
        #case if real weights
        else:
            for key,value in entry.items():    
                
                #calculate sum and normalized value
                sum_of_values=sum(entry.values())
                normalized_value=value/sum_of_values
                
                #write absolute value
                dataframe.loc[index,str(key)]=value
                
                #write normalized value
                dataframe.loc[index,str(key)+' normaliz']=normalized_value
                    
    return dataframe



def write_output_to_csv(dataframe,subdirectory=None,file_name="results.csv",string_to_write=None):
    """
    takes a dataframe and writes it as a csv file to the Results directory.
    csv_file is beeing indexed
    Seperator is ";"

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe to convert to csv.
    subdirectory : string, optional
        Should csv_file be written to a subdirectory in Results directory?
        If subdirectory does not exist, it's beeing created
        The default is None.
    file_name : String, optional
        Name of file.
        Must end on "*.csv" 
        The default is "results.csv".
    
    string_to_write : String, optional
        Optional write another string to csv file in append mode.
        The default is None.

    Raises
    ------
    ValueError
        If subdirectory does not exist.

    Returns
    -------
    None.

    """
    import os
    try:
        #Attempt to get path of current script (does not work when not running whole file)
        script_path = os.path.dirname(os.path.realpath(__file__))
    
    except NameError:
        #handle the case where '__file__' is not defined, hard code directory
        script_path = r"C:\Users\Kenny\Dropbox\Education\Uni\FU-Berlin\Inhalte\9. Semester\Bachelorarbeit\Programmierung\GitHub-Repo\LogisticRegression"
    
    # Move up one directory and navigate into Results-path
    parent_directory = os.path.abspath(os.path.join(script_path, os.pardir))
    results_directory = os.path.join(parent_directory,"Results")
    
    #if subdirectory is given, check if it's a string and if the directory exists
    if isinstance(subdirectory,str):
        try:
            results_directory= os.path.join(results_directory,subdirectory)
            if os.path.exists(results_directory):
                pass
            else:
                raise ValueError(f"the Subdirectory {results_directory} does not exist! It's beeing created.")    
        except ValueError as e:
            print(f'ValueError: {e}')
            os.makedirs(results_directory) #Create subdirectory if it does not exist    
    
    
    # Save file_path with name of csv-file
    file_path=os.path.join(results_directory,file_name)
    
    # Write dataframe as csv to file_path
    dataframe.to_csv(file_path,sep=";",index=True,index_label="index")  
    
    # Append string_to_write to csv-file
    if isinstance(string_to_write, str):
        with open(file_path,'a') as file:
            file.write(string_to_write)
    

#%% main function

def perform_grid_search(): 
    # read-in data
    data=get_data()
    
    # Filter out low irrediance values
    data=filter_data(data=data,filter_value=100) 
    
    # split data into train/test sample w. own function, scaling = True 
    x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                             test_size=0.2,scaling=True)
                                                             
    
    # Define logistic Regression Object
    logreg=LogisticRegression(multi_class="multinomial",solver="newton-cg",
                                    max_iter=500,penalty='l2',C=1.0) #C=, class_weights="balanced", "none"
    
    #%% Set Parameters to search for in Grid_search
    
    #Set Class weights
    class_weights = generate_combinations_dict(0) #currently 0 random weights, because already enough
    class_weights.append({0:1,1:1,2:1,3:1,4:1}) #no weights
    class_weights.append('balanced') #balanced
    #class_weights_t=['balanced',{0:1,1:1,2:1,3:1,4:1}] # for testing purposes
    
    #Set penalization parameter
    # penalization_param=np.logspace(-3,3,7) # penalization parameter
    penalization_param=[0.1,1,10,100,1000] # penalization parameter
    
    
    #Give over Parameters to parameter grid
    param_grid={
        'class_weight':class_weights,
        'C':penalization_param 
         }
    
    #%% Set Parameters for Grid-search
    # Define scoring metrics (for tracking and refit purposes)   
    scoring_metric={'accuracy':make_scorer(accuracy_score), 
                    'balanced_accuracy':make_scorer(balanced_accuracy_score),                
                    'macro-precision':make_scorer(precision_score,average='macro'),
                    'macro-recall':make_scorer(recall_score,average ='macro'),
                    'macro-f1_score':make_scorer(f1_score,average='macro'),
                    'weighted-precision':make_scorer(precision_score,average='weighted'),
                    'weighted-recall':make_scorer(recall_score,average ='weighted'),
                    'weighted-f1_score':make_scorer(f1_score,average='weighted')
                    }
    
    # Define k, refit and n_jobs 
    k=5
    cv=StratifiedKFold(n_splits=k,shuffle=True) # Stratified cross-validation again, k = 5
    """
    After performing this split again the dataset looks like this
    test_set = 20%
    training_set = 64% (of whole set) --> 80% of 80% original training set
    validation = 16% (of whole set) --> 20% of 80% original training set
    """
    
    refit='accuracy' #Best model shall be selected by (mean) balanced accuracy score
    n_jobs=-1 # use all CPU's later for faster performance!
    
    # Create the GridSearchCV object with k-fold cross validation 
    grid_search=GridSearchCV(logreg, param_grid, scoring=scoring_metric, cv=cv, 
                             refit=refit,n_jobs=n_jobs,verbose=2)
    
    # Fit the grid search to the data
    grid_search.fit(x_train,y_train)
    
    # Get the best parameters and corresponding model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    
    #%% Test, evaluate and print out results
    
    # Test best model on 20% test-Split    
    y_pred_best = best_model.predict(x_test)
    
    # Evaluate Best Model on test set.
    accuracy_best,report_best=evaluate_model_performance(y_test=y_test, y_pred=y_pred_best,model="Best-model Grid-search") 
    
    # Define new list for output
    csv_output=[]
    
    # Get model parameters
    csv_output.append("Model Parameters:")
    csv_output.append(best_model.get_params())
    
    # Get coefficients of model
    for i, class_coefficients in enumerate(best_model.coef_):
        csv_output.append(f"Coefficients for Class {i}: {class_coefficients}")
    
    # Append performance on test set to output
    csv_output.append(report_best)
    
    # Extract the results of Grid-search into a DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    #Append class_weights to DataFrame
    results_df=extract_class_weights(results_df)
    
    #Convert list output to one string
    string_to_write=""
    for entry in csv_output:
        string_to_write+="\n"
        string_to_write+=str(entry)
    
    #Write Dataframe and string to csv
    write_output_to_csv(results_df,subdirectory="LR",file_name="LR-Grid-search - first three class_weights_balanced_none.csv",string_to_write=string_to_write)



#%% run script
if __name__ == '__main__':
    perform_grid_search()




