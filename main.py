"""
Main-Script to run, main entry point!

"""

if __name__ == "__main__":  
    #%% Imports
    import Scripts    
    import Scripts.gridsearch.LogisticRegression as GLR
    
    #Renaming util module 
    UT=Scripts.util    
    
    
    #%%Testing
    print(UT.get_filepath(model_sd="LR",shading=True))
    
    
    #%%Run Gridsearch
    GLR.run_LR_gridsearch(shading=True)
    GLR.run_LR_gridsearch(shading=False)

    #%%Run Test-split repeaditly
    
    