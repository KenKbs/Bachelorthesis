# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:48:26 2024

@author: Kenny
"""


#%% iteration 3
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Example confusion matrix data for five models
conf_matrix_model1 = np.array([[1, 0.0000, 0.0000, 0.0000],
                               [0.0000, 2, 0.0000, 0.0000],
                               [0.0014, 0.0014, 3, 0.0000],
                               [0.0000, 0.0000, 0.0000, 4.0000],
                               ])

conf_matrix_model2 = np.array([[0.9960, 0.0010, 0.0005, 0.0005],
                               [0.0005, 0.9950, 0.0020, 0.0005],
                               [0.0020, 0.0010, 0.9930, 0.0010],
                               [0.0005, 0.0005, 0.0005, 0.9985],
                               ])

conf_matrix_model3 = np.array([[0.9972, 0.0000, 0.0000, 0.0000],
                               [0.0000, 0.9967, 0.0000, 0.0000],
                               [0.0014, 0.0014, 0.9937, 0.0000],
                               [0.0000, 0.0000, 0.0000, 1.0000],
                               ])

conf_matrix_model4 = np.array([[0.9972, 0.0000, 0.0000, 0.0000],
                               [0.0000, 0.9967, 0.0000, 0.0000],
                               [0.0014, 0.0014, 0.9937, 0.0000],
                               [0.0000, 0.0000, 0.0000, 1.0000],
                               ])

conf_matrix_model5 = np.array([[0.9972, 0.0000, 0.0000, 0.0000],
                               [0.0000, 0.9967, 0.0000, 0.0000],
                               [0.0014, 0.0014, 0.9937, 0.0000],
                               [0.0000, 0.0000, 0.0000, 1.0000],
                               ])


classes = ['0', '1', '2', '3','4']

fig, ax = plt.subplots(figsize=(12, 10))

# Annotate each cell with all five sets of scores
for i, j in itertools.product(range(conf_matrix_model1.shape[0]), range(conf_matrix_model1.shape[1])):
    # Draw a light gray rectangle as background
    rect = plt.Rectangle([j, i], 1, 1, fill=True, color='lightgray', alpha=0.5)
    ax.add_patch(rect)
    plt.text(j+0.5, i-0.2+0.5, f"{conf_matrix_model1[i, j]*100:.2f}%", ha='center', va='center', color='blue', fontsize=13)
    plt.text(j+0.5, i-0.1+0.5, f"{conf_matrix_model2[i, j]*100:.2f}%", ha='center', va='center', color='red', fontsize=13)
    plt.text(j+0.5, i+0.0+0.5, f"{conf_matrix_model3[i, j]*100:.2f}%", ha='center', va='center', color='green', fontsize=13)
    plt.text(j+0.5, i+0.1+0.5, f"{conf_matrix_model4[i, j]*100:.2f}%", ha='center', va='center', color='purple', fontsize=13)
    plt.text(j+0.5, i+0.2+0.5, f"{conf_matrix_model5[i, j]*100:.2f}%", ha='center', va='center', color='orange', fontsize=13)

# Adding class labels
plt.xticks(np.arange(len(classes)), classes,fontsize=13)
plt.yticks(np.arange(len(classes)), classes,fontsize=13)
plt.xlabel('Predicted label',fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.title('Confusion Matrix with Five Models',fontsize=25)

# Adjust plot margins to prevent overlap
# plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

# Adding a legend
blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Model 1', markerfacecolor='blue', markersize=10)
red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Model 2', markerfacecolor='red', markersize=10)
green_patch = plt.Line2D([0], [0], marker='o', color='w', label='Model 3', markerfacecolor='green', markersize=10)
purple_patch = plt.Line2D([0], [0], marker='o', color='w', label='Model 4', markerfacecolor='purple', markersize=10)
orange_patch = plt.Line2D([0], [0], marker='o', color='w', label='Model 5', markerfacecolor='orange', markersize=10)
plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch, orange_patch], loc='upper right',bbox_to_anchor=(1.15, 1))

plt.grid(True)  # Hide grid lines
plt.show()

"""
PROBLEMS:
    - AXIS LABELS NOT RIGHT; NEEDED FOR GRID; BUT SHOULD BE IN BETWEEN!
    - ORDERING OF NUMBERS IS WRONG.
    - 
"""


#%% "OLD" CODE:
from Scripts.util import load_object_from_file
import pandas as pd
import os

neutral_df= pd.DataFrame(np.full((4, 4), 100))    
neutral_df=neutral_df.values


shading=True
#load confusion matrix (from gridserach)

cm_LR=load_object_from_file("Grid-search_CM_LR.pk1",
                                     to_file="LR",shading=shading)

cm_DT=load_object_from_file("Grid-search_CM_DT.pk1",
                                     to_file="DT",shading=shading)

cm_RF=load_object_from_file("Grid-search_CM_RF.pk1",
                                     to_file="RF",shading=shading)
    
def plot_confusion_matrix(cm_LR,cm_DT,cm_RF,cm_SVM,cm_NN,
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
    #Convert to ndarray, if df is given
    if isinstance(cm_LR, pd.DataFrame):
        cm_LR=cm_LR.values
    
    if isinstance(cm_DT,pd.DataFrame):
        cm_DT=cm_DT.values
    accuracy = np.trace(cm_LR) / np.sum(cm_LR).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_LR, interpolation='nearest',cmap=cmap,alpha=1,vmax=0.9999999999,vmin=0.9999999998)
    plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm_LR = cm_LR.astype('float') / cm_LR.sum(axis=1)[:, np.newaxis]
        cm_DT = cm_DT.astype('float') / cm_DT.sum(axis=1)[:, np.newaxis]


    #plot values into Matrix
    thresh = cm_LR.max() / 1.00 if normalize else cm_LR.max() / 2 #thres first = 1.5 orginally, disable white drawing for normalized, because high class imbalance!
    for i, j in itertools.product(range(cm_LR.shape[0]), range(cm_LR.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}%".format(cm_LR[i, j]*100),
                     horizontalalignment="center",
                     color="white" if cm_LR[i, j] > thresh else "black")
            plt.text(j, i-0.1, "{:0.2f}%".format(cm_DT[i, j]*100),
                     horizontalalignment="center",
                     color="white" if cm_DT[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:0.2f}%".format(cm_LR[i, j]*100),
                     horizontalalignment="center",
                     color="white" if cm_LR[i, j] > thresh else "black")
            plt.text(j, i-0.1, "{:0.2f}%".format(cm_DT[i, j]*100),
                     horizontalalignment="center",
                     color="white" if cm_DT[i, j] > thresh else "black")

    #Adjust axis
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
    # Adding a legend
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    legend_labels = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
    legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i], markerfacecolor=colors[i], markersize=10)
    for i in range(5)
    ]
    plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.2, 1))
    
    #get current figure for saving to filepath later
    fig=plt.gcf() 
    
    #show the plot if show_plot = True
    if show_plot:
        plt.show() 
    
    #if filepath = str
    if isinstance(to_file, str):
        file_path=get_filepath(model_sd=to_file,shading=shading)
            
        #Save confusion Matrix to File as PDF    
        file_path=os.path.join(file_path,f'{title}.pdf')
        fig.savefig(file_path,format="pdf",bbox_inches='tight',pad_inches=0.1) #bbox_inches for not cutting off labels!


plot_confusion_matrix(cm_LR,cm_DT,cm_RF,None,None)

"""
make a heatmap, which is all the same color (or grey)
1) create a datframe, where all entries are equal (for example, 100, 100 100 etc.)
2) Remove Normalization thingy! Get percentage values instead! 
3) Remove Legende on the right
4) Add all 5 Models
5) Fix accuracy below

"""
