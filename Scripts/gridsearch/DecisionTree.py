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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

#Other Imports
import graphviz
import pandas as pd


#Testing, put later into function


def overfit_DT (shading=True,runs=50):
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
    #Get output path
    parent_file_path=get_filepath(model_sd="DT",shading=shading)
    file_path=parent_file_path+r'\Overfitted_tree'
    
    # Graph of DecisionTree only of last run as example
    dot_data = export_graphviz(best_model, out_file=None, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(file_path)  # Save the visualization as a file
    graph.view()  # Display the decision tree in the default viewer
    
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
    

#run function for testing:
overfit_DT(runs=5)
