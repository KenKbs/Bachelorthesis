# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:48:26 2024

@author: Kenny
"""

#%% Plotting CM
    
# Imports
from Scripts.util import (load_object_from_file,
                          plot_aggregated_confusion_matrix)

import os
import pandas as pd
import numpy as np
import itertools

#%%Plot Confusion Matrix

# Shading True (Dataset A)
shading = True

# Load confusion matrix (from gridsearch)
cm_LR = load_object_from_file("Grid-search_CM_LR.pk1",
                              to_file="LR", shading=shading)

cm_DT = load_object_from_file("Grid-search_CM_DT.pk1",
                              to_file="DT", shading=shading)

cm_RF = load_object_from_file("TestTrain_CM_RF.pk1",
                              to_file="RF", shading=shading)

cm_SVM = load_object_from_file("Grid-search_CM_SVM.pk1",
                               to_file="SVM", shading=shading)

cm_NN = load_object_from_file("Grid-search_CM_NN.pk1",
                              to_file="NN", shading=shading)

plot_aggregated_confusion_matrix(cm_LR, cm_DT, cm_RF, cm_SVM, cm_NN,
                                 title="Confusion Matrix Dataset A",to_file="FINAL",
                                 show_plot=True)

# Shading False (Dataset B)
shading = False

# Load confusion matrix (from gridsearch)
cm_LR = load_object_from_file("Grid-search_CM_LR.pk1",
                              to_file="LR", shading=shading)

cm_DT = load_object_from_file("Grid-search_CM_DT.pk1",
                              to_file="DT", shading=shading)

cm_RF = load_object_from_file("TestTrain_CM_RF.pk1",
                              to_file="RF", shading=shading)

cm_SVM = load_object_from_file("Grid-search_CM_SVM.pk1",
                               to_file="SVM", shading=shading)

cm_NN = load_object_from_file("Grid-search_CM_NN.pk1",
                              to_file="NN", shading=shading)

plot_aggregated_confusion_matrix(cm_LR, cm_DT, cm_RF, cm_SVM, cm_NN,
                                 title="Confusion Matrix Dataset B",to_file="FINAL",
                                 show_plot=True)

#%% Big tables!




