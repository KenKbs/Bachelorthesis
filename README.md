# Bachelorthesis
Bachelorthesis PV fault detection

## Logistic regression model
Logistic Regression model with grid-search optimization

## BUG FIXES
- Reworked incremental average, had a flaw in the formula
- results where not correct for train-test sample! 
- rerun train-test split for LR and DT!

## Decision Tree
Decision Tree model with Gridsearch optimization
- Output-file "DecsionTree" is the DOT representation of the decision Tree
- can be read with graphviz for example (normal .pdf files usually too large to view, therefore saved in both ways)
- Rest of files have "common" extensions
- used an "overfitted tree" to determine max-depth range to tune in gridsearch
- excluded some pruning parameters in gridsearch, because of gridsearch duration (estimated 1 month...)
- Changed that the X-data is not normalized (no z transformation) for training and testing (in comparision to the logistic regression), because not needed for convergence reasons


# CHANGES IN CURRENT VERSION
- Fixed repeaditly train-test split
- implemented Decision Trees
- "Fixed" graphviz bug, implemented a workaround if graphviz is not on system path and cannot be run from console


### To_do (still)
- Expand to other ML-models (NN, SVM, RF)


