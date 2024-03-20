"""
Main-Script to run, main entry point!

"""

if __name__ == "__main__":  
    #%% Imports
    import Scripts #for Util    
    UT=Scripts.util #Renaming util module 
    
    #Gridsearch Imports
    import Scripts.gridsearch.LogisticRegression as GLR
    import Scripts.gridsearch.DecisionTree as DT
    
    #Traintest Imports
    import Scripts.traintest.LogisticRegressiontt as TTLR
    import Scripts.traintest.DecisionTreett as TTDT

    
    #%%Testing
       
    # print(UT.get_filepath(model_sd="LR",shading=True))
        
    #%%Run Gridsearch
    
    #Logistic Regression
    # GLR.run_LR_gridsearch(shading=True)
    # GLR.run_LR_gridsearch(shading=False)
    
    
    #Decision Tree
    DT.overfit_DT(shading=True,runs=100)
    DT.overfit_DT(shading=False,runs=100)
    DT.run_DT_gridsearch(shading=True)
    DT.run_DT_gridsearch(shading=False)
    
    #DT.overfit_DT(shading=False,runs=100)
    #DT.run_DT_gridsearch(shading=False)


    #SVM

    #%%Run Test-split repeaditly
    
    #Logistic Regression
    TTLR.run_LR_traintest(shading=True,num_iterations=100)
    TTLR.run_LR_traintest(shading=False,num_iterations=100)
    
    #Decsion Tree train-test
    TTDT.run_DT_traintest(shading=True,num_iterations=100)
    TTDT.run_DT_traintest(shdaing=False,num_iterations=100)
    
    