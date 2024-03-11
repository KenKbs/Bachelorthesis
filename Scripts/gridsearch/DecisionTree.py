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
    save_object_to_file
)

#SK-learn model imports
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz



#Testing, put later into function
shading=True
 
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
            

#%%Grid search preparations

# Split data w. own fuinction, scaling = True
x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                         test_size=0.2,scaling=True)

#Define Model to tune (Logistic Regression)
dctree=DecisionTreeClassifier(splitter="best",criterion="gini")

#Set Parameter to tune

# Must Take Parameters
Criterion=["gini","entropy"] #,"log_loss"] #log loss same as shanon entropy! #Gini usually better!
max_depth=None       ##TAKE
min_samples_split=2  ##TAKE
min_samples_leaf=1   ##TAKE
max_features=None    ##TAKE
ccp_alpha=0.0 ## TAKE Post pruning cost-complexity pruning see Evernote for examples!

max_leaf_nodes=None ##MAYBE
min_impurity_decrease=0.0 ##MAYBE!

#splitter=["best","random"] #NO Use BEST!
#class_weight=None  #NO
#min_weight_fraction_leaf=0.0 #NO

"""
#Give over parameters to parameter grid to search for
param_grid={'Criterion':Criterion}

#TESTING
param_grid_t={'C':[1,1000]} #REMOVE LATER AND CHANGE FUNCTION CALL!

#%%Perform Grid_search
best_model,cv_results=perform_grid_search(x_train,y_train,logreg,param_grid_t)

#Get best_model parameters
best_model_params=best_model.get_params()
"""
#%% TESTING OF DECISION TREE!

best_model=dctree
best_model.fit(x_train,y_train)


# Graph
dot_data = export_graphviz(best_model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Save the visualization as a file
graph.view()  # Display the decision tree in the default viewer

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
"""     
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
"""

#Print out stuff
#Number nodes
num_nodes = best_model.tree_.node_count
print("Number of nodes:", num_nodes,"\n")

#depth
tree_depth = best_model.get_depth()
print("Depth of the tree:", tree_depth,"\n")

#Features
num_features = best_model.n_features_in_
print("Number of features:", num_features,"\n")

#Number of classes
num_classes = best_model.n_classes_
print("Number of classes:", num_classes,"\n")

#Feature importance
feature_importance = best_model.feature_importances_
print("Feature importance:", feature_importance,"\n")



