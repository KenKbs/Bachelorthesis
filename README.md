# Bachelorthesis
Bachelorthesis PV fault detection

## Logistic regression model
Logistic Regression model with grid-search optimization

## BUG FIXES
- Reworked incremental average, had a flow in the formula
- results where not correct for train-test sample! 
- rerun train-test split for LR and DT!

## Decision Tree
- Output-file "DecsionTree" is the DOT representation of the decision Tree
- can be read with graphviz for example (normal .pdf files usually too large to view, therefore saved in both ways)
- Rest of files have "common" extensions


# CHANGES IN CURRENT VERSION
- implemented repeaditly train-test split for Gridsearch
- implemented Decision Trees


### To_do (still)
- Expand to other ML-models (NN, SVM, RF)


