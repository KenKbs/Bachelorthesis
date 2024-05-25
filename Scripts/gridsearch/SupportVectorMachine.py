# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:44:10 2024

@author: Kenny
"""

# %% Imports


# Internal imports
from Scripts.util import (
    get_data,
    filter_data,
    generate_table,
    train_test_split_data,
    perform_grid_search,
    get_performance_metrics,
    get_confusion_matrix,
    plot_confusion_matrix,
    write_output_to_csv,
    save_object_to_file
)

# SK-learn model imports
from sklearn.svm import SVC
     
# SK-learn Preprocessing
from sklearn.model_selection import train_test_split                         

#bug-fix -Memory overflow - also does not work here without it
from joblib.parallel import parallel_config # requires joblib 1.3 or higher


# %% Gridsearch
def run_SVM_gridsearch(shading=True):
    """
    Runs Gridsearch for SupportVectorMachine and writes the best results
    to directory Results/SVM

    Parameters
    ----------
    shading : Bool, optional
        if True, includes shading class
        Else excludes it. The default is True.

    Returns
    -------
    None.

    """
    
    #Print Status
    print(f'Running Gridsearch for Support Vector Machine. Shading = {str(shading)}.')
        
    # Read-in Data
    raw_data = get_data()
    
    # filter data
    data = filter_data(raw_data, filter_value=100, shading=shading)
    
    # Print out fault distribution before and after filtering
    
    # Include shading cases
    if shading:
        generate_table(raw_data, data, "Raw", "Filtered")
    
    # Exclude shading casees
    else:
        generate_table(raw_data, data, "Raw", "Shad. excl")
    
    
    # Split data w. own fuinction, scaling = True
    x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                             test_size=0.2,
                                                             scaling=True)
    
    #reduce train size, of x_train and y_train by applying train/test again! 
    if shading:
        x_train,_,y_train,_= train_test_split(x_train, y_train, test_size = 0.75,stratify=y_train)
    """
    For dataset A, Only keep 20% of the 80% training data, so in total 20% Do with this 20% the gridsearch. Still use the 
    20% of whole data in the end for testing / verifying 
    That way, "test leackage" gets avoided, 75% of original train set get discarded (test_size = 75%)
    """
    
    # %% Define Model to tune (SVM)
    support_vm=SVC(cache_size=500, 
                  shrinking=True,
                  decision_function_shape="ovo",
                  break_ties=False)
    
    
    # %% Set Parameter to tune
    
    #Penalization_term c
    penalization_parameter=[0.0001,0.001,0.01,0.1,1,2,5,10,15,
                            25,50,75,100,
                            250,500,750,1000,2000,5000]
    #Gamma
    gamma=['scale','auto',
           0.0001,0.001,0.01,0.1,0.5,
           1,2,5,10,15,20,25,30,40,50,75,100,250,500,1000] 
    
    #Radial Basis Function (infinte dimensions)
    rbf_kernel={'C':penalization_parameter,
                'kernel':['rbf'],
                'gamma':gamma
                }
    """
    #coef0 not relevant for rbf
    coef0=[0.0] #default should be flaot, independent term
    
    #degree of polynomial - not relevant for rbf
    degree=[2]
    degree.extend(pol for pol in range (3,101)) #polynomials bis zum einhunderten Grad mit einbeziehen
    
    
    #polynomial function --> no good results, don't use
    poly_kernel={'C':penalization_parameter,
                 'kernel':['poly'],
                 'gamma':gamma,
                 'coef0':coef0,
                 'degree':degree             
                 }
    
    #linear function --> no good results, don't use
    linear_kernel={'C':penalization_parameter,
                   'kernel':['linear'],
                   }
    """
    
    # Define whole grid (list of 3 dictonaries)
    param_grid = rbf_kernel
    
    
    # %%Perform Grid_search
    with parallel_config(temp_folder='/temp',max_nbytes='4M'): #Change temporary folder to where space is (C:/temp) and maxbytes to 4 to avoid memory explosion
        best_model, cv_results = perform_grid_search(
                                 x_train, y_train, support_vm, param_grid)
    
    # Get best_model parameters
    best_model_params = best_model.get_params()
    
    # %%Performance evaluation
    
    # Predictions on test-set
    y_pred_best = best_model.predict(x_test)
    
    # Get Accuracy Score
    accuracy = get_performance_metrics(y_test, y_pred_best, only_accuracy=True)
    
    # Get f1,recall,precision etc.
    report = get_performance_metrics(y_test, y_pred_best)  # report=df
    
    # Get confusion Matrix
    cm = get_confusion_matrix(y_test, y_pred_best, normalize=False) 
    
    # %% Save Results to file / csv
    
    # plot Confusion Matrix and save to file
    plot_confusion_matrix(cm, to_file="SVM", show_plot=False, normalize=True,
                          shading=shading,
                          title=f"Grid-search ConfusionMatrix SVM shading {shading}")
    
    # save best model to file
    save_object_to_file(best_model, file_name="Best_Model",
                        to_file="SVM", shading=shading)
    
    # save confusion matrix to file:
    save_object_to_file(cm, file_name="Grid-search_CM",
                        to_file="SVM", shading=shading)
    
    # save report (f1_score etc.) to file:
    save_object_to_file(report, file_name="Grid-search_report",
                        to_file="SVM", shading=shading)
    
    # Save Grid-search results to csv_file
    write_output_to_csv(cv_results, output2=report.round(4),  # take rounded numbers of report for better overview
                        output3=best_model_params,
                        file_name="Grid-search-results_SVM",
                        to_file="SVM", shading=shading)
    
