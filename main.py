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
    import Scripts.gridsearch.RandomForest as RF
    
    #Traintest Imports
    import Scripts.traintest.LogisticRegressiontt as TTLR
    import Scripts.traintest.DecisionTreett as TTDT
    import Scripts.traintest.RandomForestt as TTRF
    

    
    #%%Testing
       
    # print(UT.get_filepath(model_sd="LR",shading=True))
        
    #%%Run Gridsearch
    
    #Logistic Regression
    # GLR.run_LR_gridsearch(shading=True)
    # GLR.run_LR_gridsearch(shading=False)
    
    
    #Decision Tree
    #DT.overfit_DT(shading=True,runs=100)
    #DT.overfit_DT(shading=False,runs=100)
    # DT.run_DT_gridsearch(shading=True)
    # DT.run_DT_gridsearch(shading=False)
    
    #DT.overfit_DT(shading=False,runs=100)
    #DT.run_DT_gridsearch(shading=False)

    #Random Forest    
    RF.run_RF_gridsearch(shading=True)
    RF.run_RF_gridsearch(shading=False)
    
    #SVM
    
    #Neural Network

    #%%Run Test-split repeaditly
    
    #Logistic Regression
    # TTLR.run_LR_traintest(shading=True,num_iterations=100)
    # TTLR.run_LR_traintest(shading=False,num_iterations=100)
    
    #Decsion Tree
    # TTDT.run_DT_traintest(shading=True,num_iterations=100)
    # TTDT.run_DT_traintest(shading=False,num_iterations=100)
    
    #Random Forest
    TTRF.run_RF_traintest(shading=True,num_iterations=100)
    TTRF.run_RF_traintest(shading=False,num_iterations=100)
    
    
    #Testing for DOT export if render with graphviz failed!
    # UT.plot_from_dot_file("Overfitted_tree", shading=True, to_file="DT",#file_path=r'C:\Users\Kenny\Dropbox\Education\Uni\FU-Berlin\Inhalte\9. Semester\Bachelorarbeit\Programmierung\GitHub-Repo\Bachelorthesis\Results\DT\Shading',
    #                       output_filename="Test123")
    