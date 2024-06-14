# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:48:26 2024

@author: Kenny
"""

#%% Plotting CM
    
# Imports
from Scripts.util import load_object_from_file
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import itertools


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


def plot_aggregated_confusion_matrix(cm_LR, cm_DT, cm_RF, cm_SVM, cm_NN,
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
    if cmap is None:
        # cmap = plt.get_cmap('Blues')
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
    plt.title(title)
    
    #Adjusting & plotting the color bar
    ax = plt.gca()
    cbar_ax = ax.inset_axes([1.05, 0, 0.03, 0.8])
    plt.colorbar(cax=cbar_ax)

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
        file_path = get_filepath(model_sd=to_file, shading=shading)
        # Save confusion Matrix to File as PDF
        file_path = os.path.join(file_path, f'{title}.pdf')
        fig.savefig(file_path, format="pdf", bbox_inches='tight', pad_inches=0.1)  # bbox_inches for not cutting off labels!


plot_confusion_matrix2(cm_LR, cm_DT, cm_RF, cm_SVM, cm_NN)
    
