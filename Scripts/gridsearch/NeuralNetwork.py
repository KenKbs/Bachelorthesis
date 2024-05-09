# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 10:44:10 2024

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
    save_object_to_file,
    get_filepath)

# SK-learn model import
from sklearn.neural_network import MLPClassifier
                              
# #Other Imports
import numpy as np
from itertools import product

#parallel configuration
from joblib.parallel import parallel_config # requires joblib 1.3 or higher


# %% Gridsearch
def run_NN_gridsearch(shading=True):
    """
    Runs Gridsearch for NeuralNetwork and writes the best results
    to directory Results/NN

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
    print(f'Running Gridsearch for Neural Network. Shading = {str(shading)}.')
    
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
    
    
    # Grid search preparations
       
    # Split data w. own fuinction, scaling = False
    x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                             test_size=0.2,
                                                             scaling=False)#???
    
    # %% Define Model to tune (NN)
    neural_network=MLPClassifier(max_iter=500) #Increased max_iter to 500
    
    
    # %% Set Parameter to tune
    """
    My Thinking:
        
        variables:
            hidden_layer_sizes ()
            activation: 'relu'
            solver: 'adam'
            alpha: '0.0001' L2 regularization term
            batch_size: 'auto' --> min(200,n_samples)
            learning_rate: 'constant'
            learning_rate_init: '0.001' --> controls step size in updating weights
            max_iter: 200 --> maybe increase this?
            tol: 1e-4 --> if not increase by at lest this, stop of learning
            (--early_stopping: False
            (--validation_fraction: percentage to set aside for validation --> not needed
            beta_1 float: 0.9    --> for adam solver
            beta_2 float: 0.999  --> for adam solver
            epsilon float: 1e-8  --> for adam solver
            n_iter_no_change: int, default =10
            
        
        On batch size:
            parameters depend on gradient computed, gradient is computed not on whole set, but
            on subset --> here the size of subset is defined
            larger = more memory and longer training time, but more stable convergence rate
            I would leave it at 200 for my case, should work, found nothing in literature that
            is problem specific
        
    """
    
    ##hidden_layer_sizes 
    #search range [2,40] [0,40]
    #Layer 1
    neurons_layer_1=[2,4,6,8,10,15,20,25,30,35,40]
    neurons_layer_1=np.array(neurons_layer_1)
    
    #Layer 2
    neurons_layer_2=[2,4,6,8,10,15,20,25,30,35,40]
    neurons_layer_2=np.array(neurons_layer_2)
    
    #List of Tuples for only the first layer
    one_hidden_layer=[(neurons,) for neurons in neurons_layer_1]
    
    #list of Tuples for first and second layer
    two_hidden_layers = list(product(neurons_layer_1, neurons_layer_2))
    
    hidden_layer_sizes=one_hidden_layer+two_hidden_layers
       
    #activation function
    activation=['relu','tanh','logistic'] #relu from Utama, tanh logistic from li
        
    #Both (Utama and default SKlearn)
    alpha=[0.0001,0.01,1,100] #4 Repräsentanten, Standard, Medium, Strong regularizaiton
    
    # TESTING
    # REMOVE LATER AND CHANGE FUNCTION CALL!
    # param_grid_t = {'hidden_layer_sizes': [(2),(15),(35),(4,35),(35,35)]}
    
    #Define parameter grid
    param_grid={'hidden_layer_sizes':hidden_layer_sizes,
                'activation':activation,
                'alpha':alpha}
    
    # %%Perform Grid_search
    with parallel_config(temp_folder='/temp',max_nbytes='4M'): #Change temporary folder to where space is (C:/temp) and maxbytes to 4 to avoid memory explosion
        best_model, cv_results = perform_grid_search(
                                 x_train, y_train, neural_network, param_grid)
    
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
    plot_confusion_matrix(cm, to_file="NN", show_plot=False, normalize=True,
                          shading=shading,
                          title=f"Grid-search ConfusionMatrix NN shading {shading}")
    
    # save best model to file
    save_object_to_file(best_model, file_name="Best_Model",
                        to_file="NN", shading=shading)
    
    # save confusion matrix to file:
    save_object_to_file(cm, file_name="Grid-search_CM",
                        to_file="NN", shading=shading)
    
    # save report (f1_score etc.) to file:
    save_object_to_file(report, file_name="Grid-search_report",
                        to_file="NN", shading=shading)
    
    # Save Grid-search results to csv_file
    write_output_to_csv(cv_results, output2=report.round(4),  # take rounded numbers of report for better overview
                        output3=best_model_params,
                        file_name="Grid-search-results_NN",
                        to_file="NN", shading=shading)
    
    
    
