# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:44:10 2024

@author: Kenny
"""

# %% Imports
from sklearnex import patch_sklearn
patch_sklearn()



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
    get_filepath
)

# SK-learn model imports
from sklearn.svm import SVC


#Extra Imports for Gridsearch done here                              
from sklearn.model_selection import train_test_split
from sklearn.preproccessing import StandardScaler

from sklearn.model_selection import GridSearchCV,StratifiedKFold

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

# #Other Imports
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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
    pass

#put later into function
shading=True

    
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

#Split data in train/test sample
# Prepare dataset and set test-size
X=data.drop('f_nv',axis=1) # Create X Matrix
Y=data['f_nv'] # Create Y-Vector
test_size=0.20 # Set Test-sample size
scaling=False
random_state=None
# Case handling for z-transformation
if scaling==True:
    scaler=StandardScaler() #Initiate Standard Scaler class of Sklearn
    X=scaler.fit_transform(X=X) # Transform data, mean = 0, variance = 1


# Split Test, Train set 
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=test_size, #put random state out later
                                                  random_state=random_state,stratify=Y) #Test,train split, Stratify by Y Vector because of imbalance of classes

# Convert Float to int to get clear classes
y_train = y_train.astype(int) 



"""
# Split data w. own fuinction, scaling = False
x_train, x_test, y_train, y_test = train_test_split_data(data=data,
                                                         test_size=0.2,
                                                         scaling=False)#???
"""
# %% Define Model to tune (SVM)
support_vm=SVC(cache_size=500, 
              shrinking=False,
              decision_function_shape="ovr",
              break_ties=False)#?


# %% Set Parameter to tune
"""
My Thinking:
    
    Fixed things:
        decision function shape (ovr / ovo)
            Only apply ovr approach, because ono approach not recommended 
            and bloats grid
        break_ties=False, because costly if enabled still do not know how useful that is...
        
        
        
        
    Need to define:
        Kernel function
        C
        gamma (for 'rbf', 'poly', and 'sigmoid', float, non-negative)
        
        coef0, (independent term) only for poly and sigmoid
        
        degree, for poly, degree of polynomial
        
    
    
    
"""

#General parameters (in every kernel)

#Penalization_term c
penalization_parameter=[0.000001,0.00001,0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                    0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,
                    0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,
                    1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,
                    10,15,20,25,30,35,40,45,50,60,70,80,90,
                    100,200,300,400,500,600,700,800,900,
                    1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,
                    3000,3250,3500,3750,4000,4250,4500,4750,
                    5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,
                    12500,15000,17500,20000,22500,25000,27500,30000,
                    40000,50000,60000,70000,80000,90000,100000,
                    200000,300000,400000,500000,600000,700000,800000,900000,
                    1000000] #Probably need a smaller grid for c...
#Gamma
gamma=['scale','auto']

#coef0
coef0=[0.0] #default should be flaot, independent term

#degree of polynomial
degree=[2]
degree.extend(pol for pol in range (3,101)) #polynomials bis zum einhunderten Grad mit einbeziehen


#Radial Basis Function (infinte dimensions)
rbf_kernel={'C':penalization_parameter,
            'kernel':['rbf'],
            'gamma':gamma
            }

#polynomial function
poly_kernel={'C':penalization_parameter,
             'kernel':['poly'],
             'gamma':gamma,
             'coef0':coef0,
             'degree':degree             
             }

#linear function
linear_kernel={'C':penalization_parameter,
               'kernel':['linear'],
               }


# Define whole grid (list of 4 dictonaries)
param_grid = [rbf_kernel, poly_kernel,linear_kernel]

# TESTING
# REMOVE LATER AND CHANGE FUNCTION CALL!
param_grid_t = {'C': [1, 0.01,2]}

# %%Perform Grid_search
with parallel_config(temp_folder='/temp',max_nbytes='4M'): #Change temporary folder to where space is (C:/temp) and maxbytes to 4 to avoid memory explosion
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
    k=5
    
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
    pre_dispatch='2*n_jobs' #see joblib documentation
    
    #Set Output (in console) during grid-search
    verbose=2 
    
    #Create GridSearchCV object
    grid_search=GridSearchCV(support_vm, param_grid_t,scoring=scoring_metric,
                             cv=cv, refit=refit,n_jobs=n_jobs,
                             pre_dispatch=pre_dispatch,verbose=verbose)
    
    #Fit the grid search to given data
    grid_search.fit(x_train,y_train)
    
    #Get Results of Search    
    best_model=grid_search.best_estimator_
    cv_results=grid_search.cv_results_   




# best_model, cv_results = perform_grid_search(
    #                          x_train, y_train, support_vm, param_grid_t)

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

# Get output path
parent_file_path = get_filepath(model_sd="SVM", shading=shading)
file_path = parent_file_path+r'\Gridsearch_SVM'

