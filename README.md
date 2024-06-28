# Bachelorthesis Performance vs. Explainability

## Introduction
Welcome to my GitHub repository for my bachelorthesis. The program aims to access the performance of five ML algorithms on the [PV fault dataset](https://github.com/clayton-h-costa/pv_fault_dataset).
The following algorithms were implemented: 
- Logistic Regression (LR)
- Decision Trees (DT)
- Random Forests (RF)
- Support Vector Machines (SVM)
- Neural Networks (NN)

For each algorithm, a grid search is conducted, and the best model is pickled and used in an iterative training and testing procedure to generate a comprehensive database for comparison. The results were aggregated, evaluated in a qualitative way, and t-tests were conducted.


## Structure of the Repository
The repository is structured as follows: 
- [Main.py](main.py) is the main script. This will run the grid search and the iterative training and testing process for each algorithm and both datasets (shading = True and False). Moreover, the [final Results](Results/FINAL/) including t-tests and aggreagation of data will be crated.
- Under [Data](Data/) you will find the raw data of the dataset linked above. 
- Under [Scripts](Scripts/), all sub-scripts are provided. The subdirectory [Gridsearch](Scripts/gridsearch/) includes all files for the gridsearch and analogeously [Traintest](Scripts/traintest/) includes all scripts for the iterative training and testing process and [visualisation.py](Scripts/analysis/Visualisation.py) is used for getting the final results. However, most functions are located in [util.py](Scripts/util.py), as they are used by all algorithms.
- The [Results](Results/) are structured according to the algorithms. Each Algorithm has its own subfolder, including subfolders shading / woShading for both datasets. Usually, results after the grid search as well as results after the iterative process are included. Pickled objects are either scikit learn ML models or pandasDataframes. The [Results/Final](Results/FINAL) Directory includes the results also used in the thesis.


## Requirements
For all code, Python version 3.11.8 has been used. As no docker or other virtualisation method was utilized, the conda environment is provided. 
- the [environment.yml](environment.yml) file can be used to set up the conda environment.
    clone the repository and create the conda environment:
  
              conda env create --name envname --file environment.yml
- the [package_conda.txt](package_conda.txt) lists all modules used. The package graphviz is optional, as there are generally problems with the installation.



