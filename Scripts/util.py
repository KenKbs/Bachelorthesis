# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:14:01 2024
IDE Spyder
@author: Kenny 
"""

#%% Imports

#System
import os
import sys #not needed yet?
from scipy.io import loadmat #for loading mat-files
import csv

#Pandas, Numpy, itertools,pickle
import pandas as pd
import numpy as np
import itertools
import pickle

#SKlearn imports / Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#SK-Metrics
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
    )

#SK-GridSearch
from sklearn.model_selection import GridSearchCV, StratifiedKFold

#Plots, graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

try: 
    import graphviz
except Exception as e:
    print("Failed to import graphviz",e)



#%% General Functions

def get_filepath(model_sd=False,shading=None):
    
    #Get current filepath of util script
    try:
        script_path=os.path.dirname(os.path.realpath(__file__))
    
    except NameError:
        #handle the case where '__file__' is not defined, hard code directory
        script_path = r"C:\Users\Kenny\Dropbox\Education\Uni\FU-Berlin\Inhalte\9. Semester\Bachelorarbeit\Programmierung\GitHub-Repo\Bachelorthesis\Scripts"
    
    #Case 1: Top Results directory -- model_sd=False (not a string)
    parent_directory=os.path.abspath(os.path.join(script_path,os.pardir))
    results_directory = os.path.join(parent_directory,"Results")

    
    #Case 2: Subdirectory of Results -- model_sd="LR","NN"..., shading = None
    if isinstance(model_sd,str):
        try:
            results_directory= os.path.join(results_directory,model_sd)
            if os.path.exists(results_directory):
                pass
            else:
                raise ValueError(f"the Subdirectory {results_directory} does not exist! It's beeing created.")    
        except ValueError as e: #Error handling, if Subdirectory not exists
            print(f'ValueError: {e}')
            os.makedirs(results_directory) #Create subdirectory if it does not exist    
        
        #Case 3: Subfolder "Shading" of specific model -- shading = True
        if shading:
            try:
                results_directory= os.path.join(results_directory,"Shading")
                if os.path.exists(results_directory):
                    pass
                else: #Error handling, if Subdirectory not exists
                    raise ValueError(f"the Subdirectory {results_directory} does not exist! It's beeing created.")    
            except ValueError as e: 
                print(f'ValueError: {e}')
                os.makedirs(results_directory) #Create subdirectory if it does not exist    
        
        #Case Handling 4:Subfolder "woShading" of specific model -- shading = False
        elif shading==False:
            try:
                results_directory= os.path.join(results_directory,"woShading")
                if os.path.exists(results_directory):
                    pass
                else: #Error handling, if Subdirectory not exists
                    raise ValueError(f"the Subdirectory {results_directory} does not exist! It's beeing created.")    
            except ValueError as e: 
                print(f'ValueError: {e}')
                os.makedirs(results_directory) #Create subdirectory if it does not exist    
    
    file_path=results_directory        
            
    return file_path
        

def get_data():
    """
    Read in Raw-Data from Data folder in parent directory
            
    Returns
    -------
    main_data : Dataframe
        Returns the raw-data from data directory.

    """
   
    # Get the directory of the current script
    try:
        #Attempt to get path of current script (does not work when not running whole file)
        script_path = os.path.dirname(os.path.realpath(__file__))
    
    except NameError:
        #handle the case where '__file__' is not defined, hard code directory
        script_path = r"C:\Users\Kenny\Dropbox\Education\Uni\FU-Berlin\Inhalte\9. Semester\Bachelorarbeit\Programmierung\GitHub-Repo\Bachelorthesis\Scripts"
    
    #Move up one directory and Navigate into data-path
    parent_directory = os.path.abspath(os.path.join(script_path, os.pardir))
    data_directory = os.path.join(parent_directory,"Data")
    
    # Save directories of data files
    amb_file_path = os.path.join(data_directory,"dataset_amb.mat")
    elc_file_path = os.path.join(data_directory,"dataset_elec.mat")
    
    # Read in files
    data_amb = loadmat(amb_file_path)
    data_elec = loadmat(elc_file_path)
    
    # Check what the keys are of matlab file
    # print(data_amb.keys())
    # print(data_elec.keys())
    
    # Save data of keys in seperate values   
    f_nv=data_amb.get('f_nv')
    irr=data_amb.get('irr')
    pvt=data_amb.get('pvt')
    idc1=data_elec.get('idc1')
    idc2=data_elec.get('idc2')
    vdc1=data_elec.get('vdc1')
    vdc2=data_elec.get('vdc2')
    
    #Transpose f_nv
    f_nv=f_nv.T
    
    # Combine all arrays and put them into a dataframe
    combined_array=np.vstack((f_nv,irr,pvt,idc1,idc2,vdc1,vdc2))
    combined_array=combined_array.T #transpose for clearness
    
    # Put everything in a DataFrame (cause I like DF more...)
    main_data=pd.DataFrame(combined_array)
    new_headings=["f_nv","irr","pvt","idc1","idc2","vdc1","vdc2"]
    main_data.columns=new_headings
    
    return main_data


def filter_data(data, filter_value=-1,shading=True):
    """
    takes raw_data, sorts it by faults and irrediance values
    and cuts out any low_irrediance values below or equal to filter_value
    Prints out the number of cases removed

    Parameters
    ----------
    data : DataFrame
        Takes a data-frame, probably from function (get_data())
    cutoff_irr_value : Float, int, optional
        Set Cutoff for low irrediance Values
        If set to <0, no filtering is done. The default is -1.(0 is alr. filtering) 
    shading : bool
        If true, include shading class, if false, exclude shading class

    Returns
    -------
    data : DataFrame
        Sorted data frame by faults and irrediance values,
        Irrediance values below cutoff have been filtered out.
    """
    # Set cutoff Value
    cutoff_irr_value=filter_value 
    
    # Sort Data first by faults and irr    
    data=data.sort_values(by=['f_nv','irr'],ascending=True)
    
    # Count entries of data before filtering
    raw_cases = data['f_nv'].count()
    
    # Filter out anything, that is below or equals cutoff value
    data=data[data['irr']>cutoff_irr_value]
    
    # Calculate removed cases
    removed_cases=raw_cases-data['f_nv'].count()
    f_removed_cases='{:,}'.format(removed_cases) #format number
            
    # Print removed cases
    print(f'\nNumber of low-irrediance cases removed: {f_removed_cases}')
    
    #When Shading = False    
    if not shading:
        pre_shaded_data=data
        data=data.drop(data[data['f_nv']==4].index) #remove shading cases
        
        #Calculate removed shading cases:
        removed_shading_cases=pre_shaded_data['f_nv'].count()-data['f_nv'].count()
        f_removed_shading_cases='{:,}'.format(removed_shading_cases) #format number
        
        #Calculate grand total cases removed:
        GT_cases=removed_cases+removed_shading_cases
        GT_cases='{:,}'.format(GT_cases) #format Cases
        
        #Print removed shading cases and overall cases:
        print(f'\nTotal number of shading cases removed: {f_removed_shading_cases}')
        print(f'\nGrandTotal cases removed: {GT_cases} ')
    
    #print Spacer and return data
    print('\n')        
    return data
    

def train_test_split_data(data,test_size=0.20,scaling=False,random_state=None):
    """
    prepares the given DataFrame to split into train and test split, using
    SK-learn function train_test_split()
    Stratified sampling is always applied.

    Parameters
    ----------
    data : DataFrame
        Prepared dataframe to perform train_test_split on.
    test_size : Float, optional
        Proportion of dataset to include in test split. Should be between
        0.0 and 1.0 The default is 0.20.
    scaling : Bool, optional
        If True, performes a z-transformation (Standardization) on the 
        X-Matrix. Feature Matrix then has 0 Mean and variance of 1
        Done because regularization l2 is sensitive to that. The default is False.
    random_state : Integer, optional
        Controls the shuffling applied to the data before applying the split. 
        Pass an int for reproducible output across multiple function calls. 
        The default is None.

    Returns
    -------
    x_train : Array
    x_test : Series
    y_train : Array
    y_test : Series
    """

    # Prepare dataset and set test-size
    X=data.drop('f_nv',axis=1) # Create X Matrix
    Y=data['f_nv'] # Create Y-Vector
    test_size=test_size # Set Test-sample size

    # Case handling for z-transformation
    if scaling==True:
        scaler=StandardScaler() #Initiate Standard Scaler class of Sklearn
        X=scaler.fit_transform(X=X) # Transform data, mean = 0, variance = 1


    # Split Test, Train set 
    x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=test_size, #put random state out later
                                                      random_state=random_state,stratify=Y) #Test,train split, Stratify by Y Vector because of imbalance of classes

    # Convert Float to int to get clear classes
    y_train = y_train.astype(int) 
    
    return x_train, x_test, y_train, y_test
    

#%% Evaluation 

def get_performance_metrics (y_test,y_pred,only_accuracy=False):
    """
    Gives back performance metrics in a dataframe 
    precision, f1, recall for each fault type OR 
    Pure accuracy score if only_accuracy = True

    Parameters
    ----------
    y_test : Series
        True values
    y_pred : Series
        Predicted Values
    only_accuracy : TYPE, optional
        if True, only the accuracy score as 
        a float is returned. The default is False.

    Returns
    -------
    accuracy : float
        
    report_df:dataframe
        dataframe containing all relevant
        metric scores as relative values (percentages)
    """
    #Get report from SK-Learn module (as dict)
    report=classification_report(y_test, y_pred,output_dict=True)

    #Remove accuracy score and save in seperate variable
    accuracy=report.pop('accuracy')
    
    #Convert to Dataframe and Transpose
    report_df=pd.DataFrame(report).T
    
    if only_accuracy:
        return accuracy
    else:
        return report_df

def get_confusion_matrix(y_test,y_pred,normalize=None):
    """
    returns confusion matrix as dataframe

    Parameters
    ----------
    y_test : Series
        set of true values.
    y_pred : Series
        set of predicted values.
    normalize : String, optional
        Takes the values 'true', 'pred', 'all'.
        Normalizes the CM to give back percentages.
        The default is None.

    Returns
    -------
    cm : Dataframe
        Confusion matrix showing wrongly and correctly predicted cases.

    """
    
    
    #Define valid normalization values and convert normalize to str
    valid_values=['true','pred','all']    
    normalize=str(normalize).lower()
    
    #Get Confusion Matrix
    #Give relative values if valid normalize argument is given
    if normalize in valid_values:
        cm=confusion_matrix(y_test, y_pred,normalize=normalize)
    
    #No normalization
    else:
        cm=confusion_matrix(y_test, y_pred)
    
    #cast ndarray to dataframe
    cm=pd.DataFrame(cm)
    
    #Return ConfusionMatrix
    return cm


def calculate_accuracy(cm):
    """
    Takes a Confusion matrix Dataframe and returns the accuracy
    First convertes it to numpy and then calculates the 
    total correct classifications

    Parameters
    ----------
    cm : DataFrame
        Confusion Matrix Dataframe

    Returns
    -------
    accuracy : Float
        Accuracy of model.
        
    """
    cm=cm.values
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    return accuracy


#%% Print out, Display, write to file
def generate_table(data1, data2=None,data1_name='data1',data2_name='data2'):
    """
    Create a table showing fault distribution with one or two dataframes

    Parameters
    ----------
    data1 : DataFrame,Series
        Puts the counts of faults (absolute and relative) into a table.
    data2 : DataFrame, optional
        Second dataframe to compare counts (optional)
    data1_name : String, optional
        Renames the first row caption of the table
    data2_name : String, optional
        Renames the sec. row caption of the table

    Returns
    -------
    str
        printed out table.

    """
    category_labels = ['0 = Normal Operation', "1 = Short Circuit","2 = Degredation",
                        '3 = Open Circuit', '4 = Shadowing'] # locally assigning again
    
    # Checks whetver input is a series or DF, if series, convert to DF
    if isinstance(data1, pd.Series):
        data1 = pd.DataFrame(data1, columns=['f_nv'])

    if data2 is not None and isinstance(data2, pd.Series):
        data2 = pd.DataFrame(data2, columns=['f_nv'])

    # Get values from DF for table.
    data1_count = data1['f_nv'].value_counts().rename(data1_name) # Absolute Values     
    data1_relcount = data1['f_nv'].value_counts(normalize=True).rename('perc.1') # Relative Value

    if data2 is not None:       
        # Manipulated Data (absolute and relative values)       
        data2_count = data2['f_nv'].value_counts().rename(data2_name)
        data2_relcount = data2['f_nv'].value_counts(normalize=True).rename('perc.2')    

        # Create Table with two columns
        cross_table = pd.concat([
            data1_count,
            data1_relcount.apply(lambda x: f"{x:.4f}"),
            data2_count,
            data2_relcount.apply(lambda x: f"{x:.4f}")
        ], axis=1, sort=True)

        # Add Total Row
        total_row = pd.DataFrame({
            data1_name: data1_count.sum(),
            'perc.1': data1_relcount.sum(),
            data2_name: data2_count.sum(),
            'perc.2': data2_relcount.sum()
        }, index=['Total'])

    else:
        # Create Table with one column
        cross_table = pd.concat([
            data1_count,
            data1_relcount.apply(lambda x: f"{x:.4f}")
        ], axis=1, sort=True)

        # Add Total Row
        total_row = pd.DataFrame({
            data1_name: data1_count.sum(),
            'perc.1': data1_relcount.sum()
        }, index=['Total'])

    cross_table.index.name = None
    cross_table.index = category_labels

    cross_table = pd.concat([cross_table, total_row])
    
    #fill NA with 0 (for example if no shading cases)
    cross_table = cross_table.fillna(0)
    
    if data2 is not None:
        #Remove trailing .0 for data2 if not same cases included 
        cross_table[data2_name] = cross_table[data2_name].astype(int).astype(str)
        cross_table[data2_name] = cross_table[data2_name].apply(lambda x: x[:-2] if x.endswith('.0') else x)
    
        # Convert the shaded and raw columns to integers and then to strings with thousands separators
        cross_table[data2_name] = cross_table[data2_name].astype(int).apply(lambda x: "{:,}".format(x))
    cross_table[data1_name] = cross_table[data1_name].astype(int).apply(lambda x: "{:,}".format(x))
    
    # Display the table
    print (f'\n\nOverview of fault distribution\n {cross_table} \n\n')


def write_output_to_csv(output,output2=None,output3=None,file_name="Results", to_file=None, shading=None):
    """
    Takes an output, converts it to a data-frame and writes it to csv file

    Parameters
    ----------
    output : dataFrame, dict, list, nparray
        Data I want to write to CSV.
    output2: dataFrame, dict, list, nparray
        additional data to be appended to csv_file
    output3: dataFrame, dict, list, nparray
        MUST BE A SINGLE LINER e.g. Best parameters of model
    file_name : Str, optional
        Name of file, without file-extension. The default is "Results".
    to_file : str, optional
        which subdirectory to write to
        e.g. "LR", or "NN" if none, wrote to top level results. The default is None.
    shading : Bool, optional
        If true, write to subfolder "Shading". 
        if False, write to subfolder "woShading"
        The default is None and writes to top level sd

    """
    #Specifiy File-path    
    file_path=get_filepath(model_sd=to_file,shading=shading)
    
    #specify File-Name
    file_name+=".csv" #Add file extension
    
    #Include full file_name to write to
    file_path=os.path.join(file_path,file_name)
    
    # if isinstance(output,(list,dict,np.array)): 
    output=pd.DataFrame(output)
    
    #Write to csv
    output.to_csv(file_path,sep=";",index=True,index_label="index")
    
    #Append additional Output2 if it's a viable type
    # if isinstance(output2,(list,dict,pd.DataFrame,np.array)):
    if output2 is not None:
        with open(file_path,"a") as file:
            file.write("\n \n") #add Blank lines
        output2=pd.DataFrame(output2)
        output2.to_csv(file_path,mode="a",sep=";",header=True)
    
    #Append additional Output3 if it's a viable type
    # if isinstance(output3,(list,dict,pd.DataFrame,np.array)):
    if output3 is not None:
        with open(file_path,"a") as file:
            file.write("\n \n") #add Blank lines
        output3=pd.DataFrame([output3])
        output3.to_csv(file_path,mode="a",sep=";",header=True,index=False)
        


def save_object_to_file(object_to_pickle,file_name,to_file=False,shading=None):
    """
    pickle an Object to a .pk1 file to load later.

    Parameters
    ----------
    object_to_pickle : Any Object

    to_file : str, False, OPTIONAL
        Model subdirectory to pickle in (LR,NN etc.)
        or write to top level (results)
        The default is False
    
    shading : Bool, optional
        if all data used (shading included) = True
        If shading is excluded = False. The default is True.
        
    file_name : Str
        Give the saved pickle file a name. Don't include file extension. 
        the to_file value e.g. "LR" and .pk1 will be added regardless. 

    Raises
    ------
    ValueError
        if Path does not exist. It will create the specific path.

    Returns
    -------
    None.

    """
    
    #get file_path
    file_path=get_filepath(to_file,shading=shading)
        
    #if subdirectory specified e.g. "LR"
    if isinstance(to_file,str):
        file_name += "_"
        file_name += to_file #name file: Best_model_MODEL-NAME 
    
    #Add File-Extension    
    file_name+=".pk1"
    
    #Save best model to pickle 
    file_path=os.path.join(file_path,file_name)
    
    #Save best-model to pickle file
    with open(file_path,'wb') as f:
        pickle.dump(object_to_pickle,f)


def load_object_from_file(file_name,file_path=None,to_file=False,shading=None):
    """
    unpickles an saved object and returns that object

    Parameters
    ----------
    file_name : String
        Name of File to load. include file extension.
    file_path : TYPE, optional
        Explicitly state directory path
        Else it's gotten from get_filepath(). The default is None.
    to_file : Str, optional
        Specify subdirectory ("LR", "NN" etc.). The default is False.
    shading : Bool, optional
        Specify shading subfolders. See get_filepath(). The default is None.

    Returns
    -------
    loaded_object : ANYTHING
        Loaded pickle object from file.

    """
    #get-Filepath if not explicitly stated
    if file_path == None:
        file_path=get_filepath(model_sd=to_file,shading=shading)
    
    #Include filename to filepath
    file_path=os.path.join(file_path,file_name)
    
    with open(file_path,"rb")as f:
        loaded_object=pickle.load(f)
    
    return loaded_object

    
        

#%% Plotting
def plot_confusion_matrix(cm,
                          target_names=None,
                          to_file=False,
                          show_plot=False,
                          shading=True,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm) or dataframe, make a nice plot
    and optionally save to file

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
                  or dataframe

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    to_file:      Bool, String
                  If True, save plot to results folder. If string
                  is given, save to subdirectory of results
                  OPTIONAL Default = False
                  
    show_plot:    Bool
                  Shows plot if True.
                  OPTIONAL Default = False
                  
    shading:  Bool
                  If True, saved into subdirectory with shading included folder
                  If False: Case = shading excluded. Save results there
                  OPTIONAL Defaullt = True

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
                  ATTENTION --> False does not work, if only relative numbers of cm as a
                  dataframe where passed over in the first place!!

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
    #Convert to ndarray, if df is given
    if isinstance(cm, pd.DataFrame):
        cm=cm.values
        
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
    
    #show the plot if show_plot = True
    if show_plot:
        plt.show() 
    
    #if filepath = str
    if isinstance(to_file, str):
        file_path=get_filepath(model_sd=to_file,shading=shading)
            
        #Save confusion Matrix to File as PDF    
        file_path=os.path.join(file_path,f'{title}.pdf')
        fig.savefig(file_path,format="pdf",bbox_inches='tight',pad_inches=0.1) #bbox_inches for not cutting off labels!

def plot_aggregated_confusion_matrix(cm_LR, cm_DT, cm_RF, cm_SVM, cm_NN,
                          target_names=None,
                          to_file=False,
                          show_plot=False,
                          title='Confusion matrix'):
    """
    given several sklearn confusion matrix (cm) or dataframe, make a nice plot
    and optionally save to file

    Arguments
    ---------
    cm_LR,cm_DT etc.:confusion matrix from sklearn.metrics.confusion_matrix
                  or dataframes for each model

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    to_file:      Bool, String
                  If True, save plot to results folder. If string
                  is given, save to subdirectory of results
                  OPTIONAL Default = False

    show_plot:    Bool
                  Shows plot if True.
                  OPTIONAL Default = False

    title:        the text to display at the top of the matrix


    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    #Only percentage values
    normalize=True
    
    # Convert to ndarray if df is given
    cm_LR = cm_LR.values
    cm_DT = cm_DT.values
    cm_RF = cm_RF.values
    cm_SVM = cm_SVM.values
    cm_NN = cm_NN.values

    # Calculate each single accuracy
    accuracy_LR = np.trace(cm_LR) / np.sum(cm_LR).astype('float')
    accuracy_DT = np.trace(cm_DT) / np.sum(cm_DT).astype('float')
    accuracy_RF = np.trace(cm_RF) / np.sum(cm_RF).astype('float')
    accuracy_SVM = np.trace(cm_SVM) / np.sum(cm_SVM).astype('float')
    accuracy_NN = np.trace(cm_NN) / np.sum(cm_NN).astype('float')

    # Calculate average accuracy
    accuracy = (accuracy_LR + accuracy_DT + accuracy_RF + accuracy_SVM + accuracy_NN) / 5
    misclass = 1 - accuracy
    
    #Define Custom color map
    colors_cmap = [(1, 1, 1),(0.9, 0.9, 1),(0.8, 0.8, 1)] 
    light_cmap = LinearSegmentedColormap.from_list("light_cmap", colors_cmap)
    
    # Custom cmap
    cmap = light_cmap

    # Normalize to percentage values
    if normalize:
        cm_LR = cm_LR.astype('float') / cm_LR.sum(axis=1)[:, np.newaxis]
        cm_DT = cm_DT.astype('float') / cm_DT.sum(axis=1)[:, np.newaxis]
        cm_RF = cm_RF.astype('float') / cm_RF.sum(axis=1)[:, np.newaxis]
        cm_SVM = cm_SVM.astype('float') / cm_SVM.sum(axis=1)[:, np.newaxis]
        cm_NN = cm_NN.astype('float') / cm_NN.sum(axis=1)[:, np.newaxis]

    # Compute std. deviation
    std_dev = np.std([cm_LR, cm_DT, cm_RF, cm_SVM, cm_NN], axis=0)

    # Normalize std. deviation for heat mapping
    norm_std_dev = (std_dev - np.min(std_dev)) / (np.max(std_dev) - np.min(std_dev))
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(norm_std_dev, interpolation='nearest', cmap=cmap, alpha=1)
    plt.title(title,fontsize=17)
    
    #Adjusting & plotting the color bar
    ax = plt.gca()
    cbar_ax = ax.inset_axes([1.05, 0, 0.03, 0.8])
    colorbar=plt.colorbar(cax=cbar_ax)
    colorbar.set_label("Normalized Std. Deviation")

    # Custom target names
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    # Plot values into Matrix
    for i, j in itertools.product(range(cm_LR.shape[0]), range(cm_LR.shape[1])):
        if normalize:
            plt.text(j, i - 0.2, "{:0.2f}%".format(cm_LR[i, j] * 100),
                     horizontalalignment="center",
                     color="black")
            plt.text(j, i - 0.1, "{:0.2f}%".format(cm_DT[i, j] * 100),
                     horizontalalignment="center",
                     color="purple")
            plt.text(j, i, "{:0.2f}%".format(cm_RF[i, j] * 100),
                     horizontalalignment="center",
                     color="green")
            plt.text(j, i + 0.1, "{:0.2f}%".format(cm_SVM[i, j] * 100),
                     horizontalalignment="center",
                     color="red")
            plt.text(j, i + 0.2, "{:0.2f}%".format(cm_NN[i, j] * 100),
                     horizontalalignment="center",
                     color="blue")
        else:
            plt.text(j, i - 0.2, "{:,}".format(cm_LR[i, j]),
                     horizontalalignment="center",
                     color="black")
            plt.text(j, i - 0.1, "{:,}".format(cm_DT[i, j]),
                     horizontalalignment="center",
                     color="purple")
            plt.text(j, i, "{:,}".format(cm_RF[i, j]),
                     horizontalalignment="center",
                     color="green")
            plt.text(j, i + 0.1, "{:,}".format(cm_SVM[i, j]),
                     horizontalalignment="center",
                     color="red")
            plt.text(j, i + 0.2, "{:,}".format(cm_NN[i, j]),
                     horizontalalignment="center",
                     color="blue")
    
    # Set x and y axis ticks to integers only (for shading=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set_yticks(np.arange(len(target_names)))
    
    # Adjust axis
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\navg_accuracy={:0.4f}; avg_misclass={:0.4f}'.format(accuracy, misclass))

    # Adding a legend
    colors = ['black', 'purple', 'green', 'red', 'blue']
    legend_labels = ['LR', 'DT', 'RF', 'SVM', 'NN']
    legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i], markerfacecolor=colors[i], markersize=10)
    for i in range(5)
    ]
    plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Get current figure for saving to filepath later
    fig = plt.gcf()

    # Show the plot if show_plot = True
    if show_plot:
        plt.show()

    # If filepath = str
    if isinstance(to_file, str):
        file_path = get_filepath(model_sd=to_file, shading=None)
        # Save confusion Matrix to File as PDF
        file_path = os.path.join(file_path, f'{title}.png')
        fig.savefig(file_path, format="png", bbox_inches='tight', pad_inches=0.1,dpi=600)  # bbox_inches for not cutting off labels!


def plot_training_time(mean_time_A_list,mean_time_B_list,model_names,
                       show_plot=False,to_file=None):
    """
    takes average training times and creates a plot and saves it to to_file
    directory if given.

    Parameters
    ----------
    mean_time_A_list : list
        mean training times for dataset A.
    mean_time_B_list : list
        mean training times for dataset B.
    model_names : list
        strings of model names for x-label
    show_plot : bool, optional
        If True, shows the plot. The default is False.
    to_file : Str, optional
        Specifiy subdirectory to safe to. The default is None.

    Returns
    -------
    None.

    """
    #Create the plot
    plt.figure(figsize=(8,6))

    #Plot training times for A
    plt.plot(model_names,mean_time_A_list,marker="o",linestyle="-",
             linewidth=2,markersize=6,color='blue',label="Dataset A")

    #Plot training times for B
    plt.plot(model_names,mean_time_B_list,marker="o",linestyle="-",
             linewidth=2,markersize=6,color='red',label="Dataset B")

    #Add Titel and labels
    plt.title('Training times for dataset A and dataset B',fontsize=14)
    plt.xlabel('ML model',fontsize=11)
    plt.ylabel('Average training time in seconds',fontsize=11)

    plt.legend(loc="best",fontsize=10)
    # plt.legend.get_frame().set_edgecolor('black')

    plt.grid(True,linestyle="--",linewidth=0.7,alpha=0.7)

    if show_plot:
        plt.show()
        
    # Get current figure for saving to filepath later
    fig = plt.gcf()
    
    # If filepath = str
    if isinstance(to_file, str):
        file_path = get_filepath(model_sd=to_file, shading=None)
        # Save confusion Matrix to File as PDF
        file_path = os.path.join(file_path, 'Training times.png')
        fig.savefig(file_path, format="png", bbox_inches='tight', pad_inches=0.1,dpi=600)  # bbox_inches for not cutting off labels!


def plot_histogram (df,title="Histogram"):
    """
    Give dataframe and Plot histogram with relative fault distribution
    also renames fault categories

    Parameters
    ----------
    df : Dataframe
        Takes dataframe as input manipulated or raw.
    title : String, OPTIONAL, Default = "Histogram"
        Give plot a title for better overview.

    Returns
    -------
    plt : matplotlib.pyplot
        Returns a plot
    """
    
    category_labels = ['0 = Normal Operation', "1 = Short Circuit","2 = Degredation",
                        '3 = Open Circuit', '4 = Shadowing']
    faults_counts=df['f_nv'].value_counts(sort=False,normalize=True)
    
    # Create a bar chart using Matplotlib
    fig, ax = plt.subplots()
    faults_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel("Categories")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(f"Histogram of Types of Faults for {title}")
    ax.set_ylim(0,0.9)
    # ax.set_xticklabels(category_labels, rotation=45, ha='right') # change category labels
    return plt


def plot_from_dot_file (file_name,file_path=None,to_file=False,shading=None,
                        output_filename=None,vector_export=True, png_export=False): 
    """
    Takes a dot string file and saves it as pdf.
    Useful, if graphviz does not run from console(because not added
                                                  to systempath)

    Parameters
    ----------
    file_name : str
        Name of dot-file to read in.
    file_path : str, optional
        Explicitly state filepath, else it's generated from
        to_file and shading. The default is None.
    to_file : Str, optional
        Name of Model (LR, DT etc.). The default is False.
    shading : Bool, optional
        if True, subfolder shading
        else subfolder woShading. The default is None.
    output_filename : str, optional
        WITHOUT File extension
        Saves file as pdf, give name of file
        standard _graphviz is added to filename as output name.
        The default is None.
    png_export : bool, optional
        if True, also exports as .png file
        might not be readable if too large
        The default is False.
    vector_export : bool, optional
        if True, also exports as .svg file (vector graph)
        no problems with close readabilty, but not a common format
        The default is True.

    Returns
    -------
    None.

    """
    
    #check if output_filename is given
    if not isinstance(output_filename, str):
        output_filename=file_name
        output_filename+='_graphviz'
    
    #get Filepath
    #get-Filepath if not explicitly stated
    if file_path == None:
        file_path=get_filepath(model_sd=to_file,shading=shading)
            
    #Create source object from DOT format string
    source = graphviz.Source.from_file(filename=file_name,directory=file_path)
    
    if vector_export:
        # Save graph as SVG (vector)
        source.render(filename=output_filename,format='svg',cleanup=False)
    
    if png_export:
        # Save graph as PNG
        source.render(filename=output_filename,format='png',cleanup=False)
    
    # Save graph as PDF
    source.render(filename=output_filename,format='pdf',cleanup=True)
    

#%% Grid search functions
def perform_grid_search(x_train,y_train,model,param_grid,k=5):
    """
    Performs grid search with predifined parameters and 
    returns best model and Results from grid search

    Parameters
    ----------
    x_train : Array
        DESCRIPTION.
    y_train : Array
        DESCRIPTION.
    model : SK-Learn-model
    param_grid : Dict
        pass over the parameters to search
        for in the Grid search
        must match tunable parameters of model
    k : int, optional
        set number of cross-fold validations
        defauult is 5.

    Returns
    -------
    best_model : Sk-learn-model
        Returns the best-sklearn model from gridsearch
        can be used for predictions etc.
    cv_results : dict of numpy ndarrays
        Callable results from grid-search
        each single iteration of Grid-search saved here

    """
    #Set Scoring Metric
    scoring_metric={'accuracy':make_scorer(accuracy_score), 
                    'balanced_accuracy':make_scorer(balanced_accuracy_score),                
                    'macro-precision':make_scorer(precision_score,average='macro'),
                    'macro-recall':make_scorer(recall_score,average ='macro'),
                    'macro-f1_score':make_scorer(f1_score,average='macro'),
                    'weighted-precision':make_scorer(precision_score,average='weighted'),
                    'weighted-recall':make_scorer(recall_score,average ='weighted'),
                    'weighted-f1_score':make_scorer(f1_score,average='weighted')
                    }
    
    # Set Gridsearch Parameters 
    
    #Set Cross-Validation (default = 5-times cv) 
    cv=StratifiedKFold(n_splits=k,shuffle=True) # Stratified sampling again
    """
    After performing this split again the dataset looks like this
    test_set = 20%
    training_set = 64% (of whole set) --> 80% of 80% original training set
    validation = 16% (of whole set) --> 20% of 80% original training set
    """
    
    #Set refit 
    refit='accuracy' #Best model shall be selected by (mean) balanced accuracy score
    
    #Set CPU-Usage
    n_jobs=-1 # all CPU Cores used
    
    #pre_dispatch
    #limit amount of jobs spawned immediatly due to memory explosion
    pre_dispatch='3*n_jobs' #see joblib documentation
    
    #Set Output (in console) during grid-search
    verbose=2 
    
    #Create GridSearchCV object
    grid_search=GridSearchCV(model, param_grid,scoring=scoring_metric,
                             cv=cv, refit=refit,n_jobs=n_jobs,
                             pre_dispatch=pre_dispatch,verbose=verbose)
    
    #Fit the grid search to given data
    grid_search.fit(x_train,y_train)
    
    #Get Results of Search    
    best_model=grid_search.best_estimator_
    cv_results=grid_search.cv_results_
    
    #Return best model and search results
    return best_model,cv_results


def get_GS_traintime (to_file,shading):
    """
    Requires get_filepath function imported
    Go to gridsearch.csv file of the specific model, find the best rank (best model)
    and give back mean fit training time.

    Parameters
    ----------
    to_file : String
        specified model ("LR","NN" etc.)
    shading : Bool
        If True, include shading, else exclude.

    Returns
    -------
    mean_fit_time : int,float
        mean_time in seconds needed on average for all 5 Cross-validation folds.

    """
    
    print('Searching for mean_fit_time of best_model in Gridsearch.csv ')
    #Get filepath
    parent_file_path=get_filepath(model_sd=to_file,shading=shading)
    #Append Name of Gridsearch.csv-file
    file_path=parent_file_path +r'\Grid-search-results_'+to_file+'.csv'
    
    #define value rank to search for
    search_value = 1 #we want to get rank = first (best model)
    
    """
    First get the column index, where the rank_test_accuracy is
    afterwards search for the first rank (1) in that column and give 
    back the mean training time (second colum)
    """

    #Open Csv_file
    with open (file_path,newline="") as file:
        
        #read in csv-file
        reader=csv.reader(file,delimiter=";")
        
        #Get column index where "rank_test_accuracy" is (depends on number of search-values)
        first_row=next(reader)
        
        #Search column index
        for index, value in enumerate(first_row):
            if value == "rank_test_accuracy":
                column_index=index
                        
        #iterate over each row for rank search
        for row in reader:
            print(f'row : {row[0]}')
            #look for rank 1 in column 14
            if search_value == int(row[column_index]):                
                #get mean fit time in secound column of matched row
                mean_fit_time=row[1] #zweite spalte = mean_fit_time                
                print(f'found rank {search_value} in row {row[0]}, giving back mean fit time {mean_fit_time}')
                
                break #break loop and return value found
                
    return mean_fit_time



#%% Test Train functions

def _is_convertible_to_float(value):
    """
    private function
    checks if value is convertible to float
    needed to extract headings of dataframe without passing
    shading argument explicitly

    Parameters
    ----------
    value : ANY
        Value to be checked.

    Returns
    -------
    bool
        if convertible = True else False.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def convert_to_srow (df,insert_value=None,extract_labels=False):
    """
    takes a dataframe and makes a single row out of it

    Parameters
    ----------
    df : DataFrame
        Data to write to a single row
    insert_value : int, float, optional
        Inserts a Value at first position of resulting string. The default is None.
    extract_labels: bool, optional
        makes the single row so it's the headings and indexes of a dataframe. 
        The default is False

    Returns
    -------
    single_row: np.Array
        All Values within one row without indexes or headings

    """
    #Extract labels if True
    if extract_labels:
        single_row = [f'{col}_{ix}' if _is_convertible_to_float(idx) else f'{col}_{idx}' for ix, idx in enumerate(df.index) for col in df.columns] 
   
    #Else Cast to single row
    else:
        single_row=df.values.flatten()
    
    """
    little wild but what happens in that if extract_labels is to check, whether 
    the index of the dataframe is still a float number or already a string
    if it's a floating number, write the current counter of the for loop into 
    the single row (so we don't have floats but clean integers)
    if it's not a float it's text so now we take the actual value of the index
    and write it to the single_row variable
    --> all this is done for casehandling shading, if we have a dataframe with indexes 
    0-3 and "average" and "weighted" or if we have cases 0-4 and "average" and "weighted" 
    """
    
    #Insert optional value in first position
    if insert_value is not None:
        single_row=np.insert(single_row,0,insert_value)
    
    return single_row


#%% ToDo

"""
- plot histogram does not look good (labelling etc.?)

"""