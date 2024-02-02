# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 9:20:59 2024

@author: Kenny
"""

#%% Imports:

import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer,accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from LR_version_2 import get_data
from LR_version_2 import filter_data
from LR_version_2 import train_test_split_data
from LR_version_2 import evaluate_model_performance
from LR_version_2 import generate_table



#%% functions


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
            
    
def plot_confusion_matrix(cm,
                          target_names=None,
                          to_file=False,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    to_file:      Bool, String
                  If True, save plot to results folder. If string
                  is given, save to subdirectory of results
                  OPTIONAL Default = False

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import os

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    #plot values into Matrix
    thresh = cm.max() / 1.00 if normalize else cm.max() / 2 #thres first = 1.5 orginally, disable white drawing for normalized, because high class imbalance!
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    #Adjust axis
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
    #get current figure for saving to filepath later
    fig=plt.gcf() 
    
    #show the plot
    plt.show() 
       
    #save plot to directory
    if isinstance(to_file,str) or to_file:
        # fig.subplots_adjust(bottom=2) #adjust cutoff value for bottom
        try:
            #Attempt to get path of current script (does not work when not running whole file)
            script_path = os.path.dirname(os.path.realpath(__file__))
        
        except NameError:
            #handle the case where '__file__' is not defined, hard code directory
            script_path = r"C:\Users\Kenny\Dropbox\Education\Uni\FU-Berlin\Inhalte\9. Semester\Bachelorarbeit\Programmierung\GitHub-Repo\LogisticRegression"
        
        # Move up one directory and navigate into Results-path
        parent_directory = os.path.abspath(os.path.join(script_path, os.pardir))
        results_directory = os.path.join(parent_directory,"Results")
        
        #if to_file = string deal with that as it is a subdirectory
        if isinstance(to_file,str):
            try:
                results_directory= os.path.join(results_directory,to_file)
                if os.path.exists(results_directory):
                    pass
                else:
                    raise ValueError(f"the Subdirectory {results_directory} does not exist! It's beeing created.")    
            except ValueError as e:
                print(f'ValueError: {e}')
                os.makedirs(results_directory) #Create subdirectory if it does not exist    
      
        file_path=os.path.join(results_directory,f'{title}.pdf')
        fig.savefig(file_path,format="pdf",bbox_inches='tight',pad_inches=0.1) #bbox_inches for not cutting off labels!
    
    

#%% main function

def perform_grid_search(): 
    # read-in data
    data=get_data()
    data_before_filter=data
    
    # Filter out low irrediance values
    data=filter_data(data=data,filter_value=100) 
    
    #Print fault distribution before and after cutoff
    generate_table(data_before_filter,data,data1_name="raw",data2_name="filtered")
    
    # split data into train/test sample w. own function, scaling = True 
    x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                             test_size=0.2,scaling=True)
                                                                 
    # Define logistic Regression Object
    logreg=LogisticRegression(multi_class="multinomial",solver="newton-cg",
                                    max_iter=500,penalty='l2',C=1.0,class_weight=None) #C=, class_weights="balanced", "none"
    """
    only solver='lbfgs' also estimates "real" multinomial model, but newton-cg better convergence
    altough it's a little slower l1 penalty not possible with newton-cg
    would need to use saga solver.
    """
    
    
    #%% Set Parameters to search for in Grid_search
    
    #Set penalization parameter
    # penalization_param=np.logspace(-3,3,7) # penalization parameter
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
                        1000000] # penalization parameter
    
    
    #Give over Parameters to parameter grid
    param_grid={'C':penalization_param}


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
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    
    #%% Test, evaluate and print out results
    
    # Make predictions on 20% test set  
    y_pred_best = best_model.predict(x_test)
    
    # Evaluate Best Model on test set.
    accuracy_best,report_best=evaluate_model_performance(y_test=y_test, y_pred=y_pred_best,model="Best-model Grid-search") 
    
    # Define new list for output, convert later to string for output
    csv_output=["\n"]
    
    # Get model parameters
    csv_output.append("Model Parameters:")
    csv_output.append(best_model.get_params())
    csv_output.append("") #for one empty line
    
    # Get coefficients of model
    csv_output.append("Coefficients of Model for each class")
    for i, class_coefficients in enumerate(best_model.coef_):
        csv_output.append(f"Coefficients for Class {i}: {class_coefficients}")
    csv_output.append("\n")
    
    # Append performance on test set to output
    csv_output.append("performance_metrics on test-set")
    csv_output.append(report_best)
    csv_output.append("")
    
    # Create confusion Matrix and append it to output:
    cm=confusion_matrix(y_test, y_pred_best)
    csv_output.append('Confusion Matrix:')
    csv_output.append(cm)
    csv_output.append("\n")
        
    #Convert csv_output list to one string
    string_to_write=""
    for entry in csv_output:
        string_to_write+="\n"
        string_to_write+=str(entry)
    
    # Extract the results of Grid-search into a DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
       
    #Write Dataframe and string to csv
    write_output_to_csv(results_df,subdirectory="LR",file_name="only l2 c.csv",string_to_write=string_to_write)

    #Create Plot of confusion matrix and save it to subdirectory LR in results directory       
    plot_confusion_matrix(cm,target_names=['Normal','shortc','degred','openc','shadw'],
                          normalize=True, to_file="LR")
  

#%% run script
if __name__ == '__main__':
    perform_grid_search()




