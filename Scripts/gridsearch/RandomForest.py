# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:44:10 2024

@author: Kenny
"""

# %% Imports
# Main package imports done in util.py

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
    load_object_from_file,  # REMOVE LATER!
    get_filepath
)

# SK-learn model imports
from sklearn.ensemble import (RandomForestClassifier
                              )
from sklearn.tree import (export_graphviz,
                          plot_tree
                          )

# #Other Imports
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# try importing graphviz; often problems, if not correctly installed
# command with conda: conda install -c conda-forge pygraphviz
try:
    import graphviz
except Exception as e:
    print("Failed to import module graphviz", e, "\n")
    print("Using Matplotlib")


# %% Gridsearch
def run_RF_gridsearch(shading=True):
    """
    Runs Gridsearch for RandomForest and writes the best results
    to directory Results/RF

    Parameters
    ----------
    shading : Bool, optional
        if True, includes shading class
        Else excludes it. The default is True.

    Returns
    -------
    None.

    """

    
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
                                                             scaling=False)
    
    # %% Define Model to tune (Decision Tree)
    rForest = RandomForestClassifier(criterion="entropy",  # CHANGE LATER AFTER RESULTS OF GS
                                     class_weight=None,
                                     min_weight_fraction_leaf=0.0,
                                     min_impurity_decrease=0.0,
                                     max_leaf_nodes=None,
                                     min_samples_leaf=1)
    
    # %% Set Parameter to tune
    """
    My Thinking:
        criterion,max_depth, min_samples_split, max_features, ccp_alpha
        taken from GS DT
        
        Would also do runs without pruned tree 
        max_depth=[None]
        min_samples_split=[2]
        max_features=[None]
        ccp_alpha = 0
        oob_score False - out of bag samples to estimate accuracy useless,
        because we have seperate test sample
        
        Left to tune
        n_estimators default = 100, should do some ranges here not too much, 
        because blows up grids
        bootstrap True / False 
        max_samples - float [0,1] percentage of samples to draw to train each base 
        estimator, when bootstrapping = True
        random_state = None, not reproduceable - no tuning here
        
        
        Doing 4 Gridsearch spaces
        Bootstrap on and off and Pruned / Unpruned trees, 
        total of 4 Grids to search for!
        
        Also account for different values for shading / noshading
        
    """
    
    # "Pruned" Values from DT Gridsearch
    
    #Values if shading Included (with shading class)
    if shading:
        max_depth = [19]
        min_samples_split = [4]
        max_features = [5]
        ccp_alpha = [0.0]
    
    else: #Values for shading = False from DT GS
        max_depth = []
        min_samples_split = []
        max_features = []
        ccp_alpha = []
    
    
    ### ADDITIONAL RF VARIABLES######
    
    # n_estimators - numbers of trees grown
    n_estimators = []  # define empty list
    n_estimators.extend(x for x in range(25, 501, 25))  # 25 to 500 in 25 steps
    
    # max_samples, proportion of whole dataset used for bootstrapping
    max_samples = [None]  # Default, whole sample used for bootstrapping
    # increment in 10% steps from 10% to 90% (100% covered with None case)
    max_samples.extend(np.arange(0.1, 1.0, 0.1))
    
    # oob_score, only useable if bootstrap = True
    # False = Default, True makes no sense, because we have a seperate validation set!
    oob_score = [False]
    
    # njobs --> should stay at default 1 (single core) to avoid over multi-processing, GS already running on all cores.
    # verbose --> no need for extra output in gridsearch
    
    
    # Define Grids to search for
    
    # Full Trees without Bootstrap
    param_grid_full_nBS = {
        # 'criterion':Criterion,
        'n_estimators': n_estimators,
        'max_depth': [None],
        'min_samples_split': [2],
        # 'min_samples_leaf':min_samples_leaf,
        'max_features': [None],
        # 'max_leaf_nodes':max_leaf_nodes,
        'ccp_alpha': [0.0],
        'bootstrap': [False]
    }
    
    # Full Trees with Bootstrap
    param_grid_full_BS = {
        # 'criterion':Criterion,
        'n_estimators': n_estimators,
        'max_depth': [None],
        'min_samples_split': [2],
        # 'min_samples_leaf':min_samples_leaf,
        'max_features': [None],
        # 'max_leaf_nodes':max_leaf_nodes,
        'ccp_alpha': [0.0],
        'bootstrap': [True],
        'max_samples': max_samples
    }
    
    # Pruned Trees without Bootstrap
    param_grid_pruned_nBS = {
        # 'criterion':Criterion,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        # 'min_samples_leaf':min_samples_leaf,
        'max_features': max_features,
        # 'max_leaf_nodes':max_leaf_nodes,
        'ccp_alpha': ccp_alpha,
        'bootstrap': [False],
    }
    
    # Pruned Trees with Bootstrap
    param_grid_pruned_BS = {
        # 'criterion':Criterion,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        # 'min_samples_leaf':min_samples_leaf,
        'max_features': max_features,
        # 'max_leaf_nodes':max_leaf_nodes,
        'ccp_alpha': ccp_alpha,
        'bootstrap': [True],
        'max_samples': max_samples
    }
    
    # Define whole grid (list of 4 dictonaries)
    param_grid = [param_grid_full_nBS, param_grid_full_BS,
                  param_grid_pruned_nBS, param_grid_pruned_BS]
    # TESTING
    # REMOVE LATER AND CHANGE FUNCTION CALL!
    param_grid_t = {'criterion': ['gini', 'entropy']}
    
    # %%Perform Grid_search
    best_model, cv_results = perform_grid_search(
        x_train, y_train, rForest, param_grid_t)
    
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
    plot_confusion_matrix(cm, to_file="RF", show_plot=False, normalize=True,
                          shading=shading,
                          title=f"Grid-search ConfusionMatrix RF shading {shading}")
    
    # save best model to file
    save_object_to_file(best_model, file_name="Best_Model",
                        to_file="RF", shading=shading)
    
    # save confusion matrix to file:
    save_object_to_file(cm, file_name="Grid-search_CM",
                        to_file="RF", shading=shading)
    
    # save report (f1_score etc.) to file:
    save_object_to_file(report, file_name="Grid-search_report",
                        to_file="RF", shading=shading)
    
    # Save Grid-search results to csv_file
    write_output_to_csv(cv_results, output2=report.round(4),  # take rounded numbers of report for better overview
                        output3=best_model_params,
                        file_name="Grid-search-results_RF",
                        to_file="RF", shading=shading)
    
    # Get output path
    parent_file_path = get_filepath(model_sd="RF", shading=shading)
    file_path = parent_file_path+r'\Gridsearch_RF'
    
    # %% Plot and save decision Tree
    
    # Extract feature and class names
    fn = x_test.columns.tolist()
    cn = y_test.unique().tolist()
    cn = sorted(cn)
    
    # cast class names into string
    cn = [str(Float) for Float in cn]
    
    # Plot one Tree of RF
    try:
        dot_data = export_graphviz(best_model.estimators_[0], out_file=None, filled=True,
                                   rounded=True, special_characters=True,
                                   feature_names=fn, class_names=cn)
        graph = graphviz.Source(dot_data)
        graph.render(file_path, format="pdf")  # Save the visualization as a file
        graph.view()  # Display the decision tree in the default viewer
    
    # if Error, render with matplotlib, first tree of forest
    except Exception as e:
        print("Error occured while rendering with graphviz:", e, "\n\n")
        print("Attempt to render using Matplotlib")
        file_path += r'.pdf'
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(150, 50), dpi=200)
        plot_tree(best_model.estimators_[0], feature_names=fn,
                  class_names=cn, filled=True)
        plt.savefig(file_path)  # Save the plot to pdf-file
    
    """
    graphviz does not work from console, if not added to system path
    because adding to system path is kinda inconvenient, did a try except statement
    with a different visualisation tool.
    """
