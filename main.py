"""
Main-Script to run, main entry point!

"""

if __name__ == "__main__":  
    #%% Imports
    import Scripts    
    import Scripts.gridsearch.LogisticRegression as GLR
    import Scripts.traintest.LogisticRegressiontt as TTLR
    
    #%%Testing
    #Renaming util module 
    UT=Scripts.util        
    #print(UT.get_filepath(model_sd="LR",shading=True))
        
    #%%Run Gridsearch
    #GLR.run_LR_gridsearch(shading=True)
    #GLR.run_LR_gridsearch(shading=False)

    #%%Run Test-split repeaditly
    #TTLR.run_LR_traintest(shading=True,num_iterations=100)
    TTLR.run_LR_traintest(shading=False,num_iterations=100)
    
    
    
    