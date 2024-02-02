# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:23:48 2024

@author: Kenny
"""

#%% imports --> moved to main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for Histogram






#%% Functions

def get_data():
    """
    Read in Raw-Data from Data folder in parent directory

    Returns
    -------
    main_data : Dataframe
        Returns the raw-data from data directory.

    """
    import os
    from scipy.io import loadmat #for loading mat files
    # Get the directory of the current script
    try:
        #Attempt to get path of current script (does not work when not running whole file)
        script_path = os.path.dirname(os.path.realpath(__file__))
    
    except NameError:
        #handle the case where '__file__' is not defined, hard code directory
        script_path = r"C:\Users\Kenny\Dropbox\Education\Uni\FU-Berlin\Inhalte\9. Semester\Bachelorarbeit\Programmierung\GitHub-Repo\LogisticRegression"
    
    # Move up one directory and navigate into data-path
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


def get_summarystats (df):
    """
    prints out info stats and returns summary data

    Parameters
    ----------
    df : Dataframe
        takes Dataframe as input.

    Returns
    -------
    summary_stats : Dataframe
        Returns summary data as variable for better inspection.

    """   
    info_stats=df.info
    summary_stats = df.describe()
    string_to_print =f'Here is the general info of the df {df}: \n{info_stats}\n \n \n'
    string_to_print += f'The summary stats are as follows: \n{summary_stats}\n '
    string_to_print += 'The summary stats also have been returned as a variable to inspect it'
    print(string_to_print)
    return(summary_stats)


def plot_histogram (df,title):
    """
    Plot histogram with relative fault distribution
    also renames fault categories

    Parameters
    ----------
    df : Dataframe
        Takes dataframe as input manipulated or raw.
    title : String
        Give plot a title for better overview.

    Returns
    -------
    plt : matplotlib.pyplot
        Returns a plot
    """
    global category_labels
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
        # Manipulated Data
        data2_count = data2['f_nv'].value_counts().rename(data2_name)
        data2_relcount = data2['f_nv'].value_counts(normalize=True).rename('perc.2')

        # Create Table with two columns
        cross_table = pd.concat([
            data1_count,
            data1_relcount.apply(lambda x: f"{x:.6f}"),
            data2_count,
            data2_relcount.apply(lambda x: f"{x:.6f}")
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
            data1_relcount.apply(lambda x: f"{x:.6f}")
        ], axis=1, sort=True)

        # Add Total Row
        total_row = pd.DataFrame({
            data1_name: data1_count.sum(),
            'perc.1': data1_relcount.sum()
        }, index=['Total'])

    cross_table.index.name = None
    cross_table.index = category_labels

    cross_table = pd.concat([cross_table, total_row])

    # Display the table
    print (f'\n\nOverview of fault distribution\n {cross_table} \n\n')


def evaluate_model_performance (y_test,y_pred,model="model",
                                shortend=False,not_include_results=False,
                                append_dict=None):
    """
    prints and returns standard performance metrics
    if a dictonary is give, accuracy is appended to dict

    Parameters
    ----------
    y_test : Series
        Numpy Series of test set
    y_pred : Series
        Numpy Series of predictions made my model to evaluate
        y_test,y_pred must be of same length
    model : String, optional
        Specifiy the Model name for Output purposes. The default is "model".
    shortend : Bool, optional
        If True, prints shortend output (only accuracy). The default is False.
    not_include_results : Bool, optional
        If True, does not return accuracy and report as variables. The default is False.
    append_dict: dictonary, optional
        if specified, Accuracy score gets appended to dictonary, with model_name
        if entry already exists, get overwritten
        

    Returns
    -------
    accuracy : float64
        Accuracy Score of model
    report : String
        Inludes precision, recall, accuracy.

    """
    #Imports 
    from sklearn.metrics import accuracy_score, classification_report #import again locally for function export to other scripts
    
    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    string_to_print = ""

    # Case handling for shortened Printing
    if not shortend:
        string_to_print += f'\n Evaluation of {model}:\n'
        string_to_print += f'\n Accuracy of {model}: {accuracy:.2f} \n\n'
        string_to_print += f'Classification Report of {model}:\n{report}\n'
    else:
        string_to_print += f'\n Accuracy of {model}: {accuracy:.2f} \n\n'  # only prints Accuracy Score

    # Add divider to String and print it
    string_to_print += "*" * 50
    string_to_print += f'End of {model}'
    print(string_to_print)

    # Update Dictionary with model entry and case handling for return
    if append_dict is not None and isinstance(append_dict, dict):
        append_dict[model] = accuracy*100 #times 100 to get percentages for better readability
        
        #Case handling for when Dict is given:
        if not not_include_results:
            return accuracy, report, append_dict #dict given & not_include_results = False
        
        elif not_include_results:
            return append_dict #dict given & not_include_results= True
   
    #case handling for no dict given
    else: #No dict given
        if not not_include_results: #not_include_results = False
            return accuracy, report  #only return report and accuracy
        
        elif not_include_results: #not_include_results = True
            return None
    


def filter_data (data,filter_value=0):
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
        If set to 0, no filtering is done. The default is 0. 

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
    
    # Print removed cases
    print(f'\nTotal number of cases removed: {removed_cases}')
       
    return data
    

def train_test_split_data (data,test_size=0.20,scaling=False,random_state=None):
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
    
    # Imports locally
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
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


#%% Main Function
def main():
    
    from sklearn.linear_model import LogisticRegression
    import statsmodels.api as sm
    import seaborn as sns # for Heatmap
    
    #%% clear all variables

    from IPython import get_ipython
    get_ipython().magic('reset -sf')


    #%% Read in raw_data and first look on raw Data
    raw_data=get_data()
    
    # raw_data=raw_data.sort_values(by=['f_nv','irr'],ascending=True) #sort by error and irrediance
    raw_data_summary=get_summarystats(raw_data)
    
    # Plot fault distribution for Raw-data
    plot_histogram(raw_data,'Raw-data').show() 
    
    
    #%% Filter out data with low irradiance
    
    # Filter data
    data=filter_data(raw_data,filter_value=200)
    
    # Get summary stats
    data_summary=get_summarystats(data)
    
    # Plot fault distribution for filtered data
    plot_histogram(data, "manipulated data") 
    
    # Create Table showing fault distribution
    generate_table(raw_data,data2=data,data1_name="raw_data",data2_name="manip.data")
    
    
    #%% Data preparance for Logistic Regression
    
    x_train, x_test, y_train, y_test = train_test_split_data(data,test_size=0.20,
                                                             scaling=False,
                                                             random_state=42)
    
    # #Split dataset into 80/20% and then perform a grid search with k-fold validation on the 80% with a grid search
    # X=data.drop('f_nv',axis=1) # Create X Matrix
    # Y=data['f_nv'] # Create Y-Vector
    # test_size=0.20 # Set Test-sample size
    
    # # Split Test, Train set
    # x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=test_size, #put random state out later
    #                                                   random_state=42,stratify=Y) #Test,train split, Stratify by Y Vector because of imbalance of classes
    
    # # Convert Float to int to get clear classes
    # y_train = y_train.astype(int) 
    
    # look at fault distribution of train/test data --> Stratisfying works
    generate_table(y_train,y_test,data1_name='y_train',data2_name='y-test') 
    
    
    #%% Logistic Regression Statmodels
    
    # Plot correlation matrix
    correlation_matrix=data.corr() #Drop idc1, because high correlation!!! And faults only introduced in 2
    sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm') # Draw correlation heatmap
    plt.title('Heatmap for correlation of data')
    plt.show 
    
    # Pepare Data: add constant + drop idc1
    x_train_sm = sm.add_constant(x_train) # Add constant
    x_test_sm = sm.add_constant(x_test)
    x_train_sm = x_train.drop('idc1',axis=1) #Could also drop vdc1
    x_test_sm = x_test.drop('idc1',axis=1) # dropping idc1, because if no variable dropped, no clear results!
    
    #declare y- test and train set for sm-regression
    y_train_sm=y_train
    y_test_sm=y_test
    
    # Fit multinomial regression model
    logit_model = sm.MNLogit(y_train_sm, x_train_sm)
    result = logit_model.fit()
    
    # Display summary of the model
    stats1=result.summary()
    stats2=result.summary2()
    print(stats1)
    print(stats2)
    
    # Test model
    y_pred_probs = result.predict(x_test_sm)
    y_pred_sm = np.argmax(y_pred_probs.values, axis=1)
    
    accuracy_overview={} #Create dictonary to compare accuracy scores of models
    
    # Evaluate Model
    accuracy_sm,report_sm,accuracy_overview=evaluate_model_performance(y_test_sm, y_pred_sm,
                                                                       model="StatsmodelSm",shortend=True,
                                                                       not_include_results=False,
                                                                       append_dict=accuracy_overview) 
    
    
    #%% Logistic Regression OVR using SKLearn
    
    #Define Logistic Regression Object
    logreg_ovr = LogisticRegression(multi_class="ovr",solver="lbfgs",
                                    max_iter=1000,penalty=None)                                
    
    # Train model
    logreg_ovr.fit(x_train,y_train) 
    
    # Test model
    y_pred_ovr = logreg_ovr.predict(x_test)
    
    # Evaluate model: 
    accuracy_OVR,report_OVR,accuracy_overview=evaluate_model_performance(y_test, y_pred_ovr,
                                                                       model="SKLearnOVR",shortend=True,
                                                                       not_include_results=False,
                                                                       append_dict=accuracy_overview)     
    
    # Print the model parameters
    print("Model Parameters:")
    print(logreg_ovr.get_params())
    
    # Print other model information
    print("Model Information:")
    print(logreg_ovr)
    
    for i, class_coefficients in enumerate(logreg_ovr.coef_):
        print(f"Coefficients for Class {i}: {class_coefficients}")
    
    
    
    #%% Logistic Regression MN with SKlearn and penalization term (l2)
    
    #Define Logistic Regression Object
    logreg_skmn = LogisticRegression(multi_class="multinomial",solver="newton-cg",
                                    max_iter=1000,penalty='l2',C=1.0)                            
    
    # Train model
    logreg_skmn.fit(x_train,y_train) 
    
    # Test model
    y_pred_skmn = logreg_skmn.predict(x_test)
    
    
    #Evaluate Model
    accuracy_SKMN,report_SKMN,accuracy_overview=evaluate_model_performance(y_test, y_pred_skmn,
                                                                       model="SKLearnMN w.l2 ",shortend=True,
                                                                       not_include_results=False,
                                                                       append_dict=accuracy_overview) 
    
    # Print the model parameters
    print("Model Parameters:")
    print(logreg_skmn.get_params())
    
    # Print other model information
    print("Model Information:")
    print(logreg_skmn)
    
    for i, class_coefficients in enumerate(logreg_skmn.coef_):
        print(f"Coefficients for Class {i}: {class_coefficients}")
    
    
    
    #%% Multinomial logistic regression WITH Normalized data
    
    x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split_data(data,
                                                                                         test_size=0.20,
                                                                                         scaling=True,
                                                                                         random_state=42)
    
    
    #%% OVR again: 
    
    # Initiate Model    
    logreg_ovr_scaled = LogisticRegression(multi_class="ovr",solver="lbfgs",
                                    max_iter=500,penalty=None)   
    
    # Train model
    logreg_ovr_scaled.fit(x_train_scaled,y_train_scaled) 
    
    # Test model
    y_pred_ovr_scaled = logreg_ovr_scaled.predict(x_test_scaled)
    
    # Evaluate Model
    accuracy_OVR_scaled,report_OVR_scaled,accuracy_overview=evaluate_model_performance(y_test_scaled, y_pred_ovr_scaled,
                                                                       model="SKLearnOVR - SCALED",shortend=True,
                                                                       not_include_results=False,
                                                                       append_dict=accuracy_overview) 
    
    # Print the model parameters
    print("Model Parameters:")
    print(logreg_ovr_scaled.get_params())
    
    # Print other model information
    print("Model Information:")
    print(logreg_ovr_scaled)
    
    for i, class_coefficients in enumerate(logreg_ovr_scaled.coef_):
        print(f"Coefficients for Class {i}: {class_coefficients}")
    
    
    #%% Multinomial again with l2 penalization:
    
    #Define Logistic Regression Object
    logreg_skmn_scaled = LogisticRegression(multi_class="multinomial",solver="newton-cg",
                                    max_iter=500,penalty='l2',C=1.0)                            
    
    # Train model
    logreg_skmn_scaled.fit(x_train_scaled,y_train_scaled) 
    
    # Test model
    y_pred_skmn_scaled = logreg_skmn_scaled.predict(x_test_scaled)
    
    # Evaluate Model
    accuracy_SKMN_Scaled,report_SKMN_Scaled,accuracy_overview=evaluate_model_performance(y_test_scaled, y_pred_skmn_scaled,
                                                                       model="SKLearnMN w.l2 - SCALED",shortend=True,
                                                                       not_include_results=False,
                                                                       append_dict=accuracy_overview) 
    
    # Print the model parameters
    print("Model Parameters:")
    print(logreg_skmn_scaled.get_params())
    
    # Print other model information
    print("Model Information:")
    print(logreg_skmn_scaled)
    
    for i, class_coefficients in enumerate(logreg_skmn_scaled.coef_):
        print(f"Coefficients for Class {i}: {class_coefficients}")
    
        



#%% run main
   
if __name__ == '__main__':
    main()

    