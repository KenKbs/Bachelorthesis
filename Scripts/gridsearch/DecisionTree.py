# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:44:10 2024

@author: Kenny
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:21:50 2024

@author: Kenny
"""
#%% Imports 
#Main package imports done in util.py

#Internal imports
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
    get_filepath
)

#SK-learn model imports
from sklearn.tree import (DecisionTreeClassifier,
                          export_graphviz,
                          plot_tree)

#Other Imports
import pandas as pd
import numpy as np

#try importing graphviz problems, if not correctly installed
try:   
    import graphviz
except Exception as e:
    print("Failed to import module graphviz",e,"\n")
    print("Using Matplotlib")

import matplotlib.pyplot as plt




#%% Gridsearch
def run_DT_gridsearch(shading=True):
    #Read-in Data
    raw_data=get_data()
    
    #filter data
    data=filter_data(raw_data,filter_value=100,shading=shading)
    
    #Print out fault distribution before and after filtering
    
    #Include shading cases
    if shading:
        generate_table(raw_data,data,"Raw","Filtered")
    
    #Exclude shading casees
    else:
        generate_table(raw_data,data,"Raw","Shad. excl")
                
    
    #%%Grid search preparations
    
    # Split data w. own fuinction, scaling = True
    x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                             test_size=0.2,scaling=True)
    
    # Define Model to tune (Decision Tree)
    dctree=DecisionTreeClassifier(splitter="best",class_weight=None,
                                  min_weight_fraction_leaf=0.0,
                                  min_impurity_decrease=0.0,
                                  max_leaf_nodes=None,
                                  min_samples_leaf=1)
    
    # Set Parameter to tune
    
    # Criterion
    Criterion=["gini","entropy"] #,"log_loss"] #log loss same as shanon entropy! #Gini usually better!
    
    
    #max_depth 
    max_depth=[None] #- prevent overfitting, depth of tree
    #append values from 1 to 41 in steps = 1 to list
    max_depth.extend(x for x in range (1,42,1))
    
    
    #max_leaf_nodes
    # max_leaf_nodes=[None] # maximum number of leaf nodes
    
    #append values from 50 to 3050 in 50 steps
    # max_leaf_nodes.extend(x for x in range (500,3001,100))
    
    
    #min_samples_split  
    min_samples_split=[2] #minimum number of samples requiered for splitting a node
    
    #append values from   
    min_samples_split.extend(x for x in range(3,26,1))
   
    
    #min_samples_leaf 
    # min_samples_leaf=[1] # minimum number of samples required in last decision leafs
    # min_samples_leaf.extend(x for x in range(2,26,1))
    
    
    #max_features
    max_features=[None] # - randomly select x features, from these x features, determine which is best to split and do the splitting
    
    #append 1 to 6, because 7 = maxfeatures, this case covered with None
    max_features.extend(x for x in range (1,7,1))    
    
    
    #ccp_alpha 
    ccp_alpha=[0.0] ## minimal cost pruning, after fitting complete prune tree - small alpha values!!!
    
    #append 0.005 to 0.501 in 0.005 steps
    ccp_alpha.extend(np.arange(0.005,0.501,0.005))
    
    
    
    #Give over parameters to parameter grid to search for
    param_grid={'criterion':Criterion,
                'max_depth':max_depth,
                'min_samples_split':min_samples_split,
                # 'min_samples_leaf':min_samples_leaf,
                'max_features':max_features,
                # 'max_leaf_nodes':max_leaf_nodes,
                'ccp_alpha':ccp_alpha
                }
    
    #TESTING
    param_grid_t={'criterion':Criterion} #REMOVE LATER AND CHANGE FUNCTION CALL!
    
    #%%Perform Grid_search
    best_model,cv_results=perform_grid_search(x_train,y_train,dctree,param_grid_t)
    
    #Get best_model parameters
    best_model_params=best_model.get_params()
    
    
    #%%Performance evaluation
    
    #Predictions on test-set
    y_pred_best=best_model.predict(x_test)
    
    #Get Accuracy Score
    accuracy=get_performance_metrics(y_test, y_pred_best,only_accuracy=True)
    
    #Get f1,recall,precision etc.
    report=get_performance_metrics(y_test, y_pred_best) #report=df
    
    #Get confusion Matrix
    cm=get_confusion_matrix(y_test, y_pred_best,normalize=False)
        
    
    #%% Save Results to file / csv 
    
    #plot Confusion Matrix and save to file
    plot_confusion_matrix(cm,to_file="DT",show_plot=True,normalize=True,
                          shading=shading,
                          title=f"Grid-search ConfusionMatrix DT shading {shading}")
        
    #save best model to file
    save_object_to_file(best_model,file_name="Best_Model",
                        to_file="DT",shading=shading)
    
    #save confusion matrix to file:
    save_object_to_file(cm,file_name="Grid-search_CM",
                        to_file="DT",shading=shading)
    
    #save report (f1_score etc.) to file:
    save_object_to_file(report,file_name="Grid-search_report",
                        to_file="DT",shading=shading)
    
    #Save Grid-search results to csv_file
    write_output_to_csv(cv_results,output2=report.round(4), #take rounded numbers of report for better overview
                        output3=best_model_params,
                        file_name="Grid-search-results_DT",
                        to_file="DT",shading=shading)
    
    #Get output path
    parent_file_path=get_filepath(model_sd="DT",shading=shading)
    file_path=parent_file_path+r'\Gridsearch_tree'
    
    # Plot and save decision Tree
    # With Graphviz
    try: 
        dot_data = export_graphviz(best_model, out_file=None, filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(file_path, format="pdf")  # Save the visualization as a file
        graph.view()  # Display the decision tree in the default viewer
    
    #if Error, render with matplotlib
    except Exception as e:
        print("Error occured while rendering with graphviz:",e,"\n\n")
        print("Attempt to render using Matplotlib")       
        file_path+=r'.pdf'
        plt.figure(figsize=(120, 60))  # Set the figure size
        plot_tree(best_model, filled=True, rounded=True)#, feature_names=feature_names, class_names=target_names)  # Plot the decision tree
        plt.savefig(file_path)  # Save the plot to a file
    
    """
    graphviz does not work from console, if not added to system path
    because adding to system path is kinda inconvenient, did a try except statement
    with a different visualisation tool.
    """

#%%Overfit Decision Tree (just default values)
def overfit_DT (shading=True,runs=100):
    #%% Read in and filter data
    
    #Read-in Data
    raw_data=get_data()
    
    #filter data
    data=filter_data(raw_data,filter_value=100,shading=shading)
    
    #Print out fault distribution before and after filtering
    
    #Include shading cases
    if shading:
        generate_table(raw_data,data,"Raw","Filtered")
    
    #Exclude shading casees
    else:
        generate_table(raw_data,data,"Raw","Shad. excl")
                
    
    #%%Grid search preparations - dertimine parameter ranges to tune:
    
    #Starting values
    #define empty lists
    num_nodes_all=[]
    tree_depth_all=[]
    num_features_all=[]
    accuracy_all=[]   
    
    #5 runs with "overfitted" Tree to see stats of that algorithm
    for run in range (runs):
        # Split data w. own fuinction, scaling = True
        x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                                 test_size=0.2,scaling=True)
        
        #Define Model to tune (Logistic Regression)
        dctree=DecisionTreeClassifier(splitter="best",criterion="gini",
                                      max_depth=None,min_samples_split=2,
                                      min_samples_leaf=1,max_features=None,
                                      ccp_alpha=0.0,max_leaf_nodes=None,
                                      min_impurity_decrease=0.0)
        
        #%% TESTING OF DECISION TREE to determine which paramter ranges to tune!
        
        best_model=dctree
        best_model.fit(x_train,y_train)
           
        #Predictions on test-set
        y_pred_best=best_model.predict(x_test)
                            
        #Print out stuff
        #current run    
        print(f"Current run: {run+1} \n")
        
        #Number nodes
        num_nodes = best_model.tree_.node_count
        print("Number of nodes:", num_nodes,"\n")
        num_nodes_all.append(num_nodes)
        
        #depth
        tree_depth = best_model.get_depth()
        print("Depth of the tree:", tree_depth,"\n")
        tree_depth_all.append(tree_depth)
        
        #Features
        num_features = best_model.n_features_in_
        print("Number of features:", num_features,"\n")
        num_features_all.append(num_features)
        
        #Number of classes
        num_classes = best_model.n_classes_
        print("Number of classes:", num_classes,"\n")
        
        #Feature importance
        feature_importance = best_model.feature_importances_
        print("Feature importance:", feature_importance,"\n")
            
        #Get Accuracy Score
        accuracy=get_performance_metrics(y_test, y_pred_best,only_accuracy=True)
        accuracy_all.append(accuracy)
        
        print ("*"*40)
    
    #%% Save Results to file / csv 
    
    #Get confusion Matrix only for last run as example
    cm=get_confusion_matrix(y_test, y_pred_best,normalize=False)
    
    #plot last Confusion Matrix and save to file
    plot_confusion_matrix(cm,to_file="DT",show_plot=True,normalize=True,
                          shading=shading,
                          title=f"Overfitted ConfusionMatrix DT shading {shading}")
    
    #create Dataframe and write to csv
    dt_overfitted=pd.DataFrame()
    
    #Append important values
    dt_overfitted["Number of nodes:"]=num_nodes_all
    dt_overfitted["Depth of the tree:"]=tree_depth_all
    dt_overfitted["Number of features:"]=num_features_all
    dt_overfitted["Accuracy"]=accuracy_all
    
    #Append Max-Values
    max_values = dt_overfitted.max()
    dt_overfitted.loc[len(dt_overfitted)] = max_values
    
    #Re-adjust index
    dt_overfitted.rename({dt_overfitted.index[-1]:"Max"},inplace=True)
    
    #Get model Params:
    model_params=best_model.get_params()
    
    #write Dataframe to csv
    write_output_to_csv(dt_overfitted,output3=model_params,
                        file_name="Overfitted_tree",
                        to_file="DT",shading=shading)
    #Get output path
    parent_file_path=get_filepath(model_sd="DT",shading=shading)
    file_path=parent_file_path+r'\Overfitted_tree'
    
    # Plot and save decision Tree only of last run as example
    # With Graphviz
    try: 
        dot_data = export_graphviz(best_model, out_file=None, filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(file_path, format="pdf")  # Save the visualization as a file
        graph.view()  # Display the decision tree in the default viewer    
    
    #With matplotlib if it fails
    except Exception as e:
        print("Error occured while rendering with graphviz:",e,"\n\n")
        print("Attempt to render using Matplotlib")       
        file_path+=r'.pdf'
        plt.figure(figsize=(120, 60))  # Set the figure size
        plot_tree(best_model, filled=True, rounded=True)#, feature_names=feature_names, class_names=target_names)  # Plot the decision tree
        plt.savefig(file_path)  # Save the plot to a file
    
    
#run function for testing:
#overfit_DT(shading=False,runs=2)
#run_DT_gridsearch(shading=True)
