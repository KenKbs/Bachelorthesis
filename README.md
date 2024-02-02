# Bachelorthesis
Bachelorthesis PV fault detection

## Logistic regression model
Logistic Regression model with grid-search optimization

# CHANGES IN VERSION 3 LR Grid search
- Added Confusion Matrix to output as PDF
- Removed Class weights as search parameter from Grid-search
- fixed bug, that data was not filtered out by low irrediance#
- added penalization paramters c to search for 

### To_do (still)
- Remove shading class on data for comparison of performance with / without shading class
- save best settings for model to repeat process of training and fitting
- generate loop to repeaditly train and test best model for both instances (with / without shading)
- Think of a way:
  - Which performance metrics to store
  - How to store generated data, in which format
  - How to easily access generated data later for statistical test for evaluation
