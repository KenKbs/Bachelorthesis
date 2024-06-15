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
    import Scripts.gridsearch.SupportVectorMachine as SVM
    import Scripts.gridsearch.NeuralNetwork as NN
    
    #Traintest Imports
    import Scripts.traintest.LogisticRegressiontt as TTLR
    import Scripts.traintest.DecisionTreett as TTDT
    import Scripts.traintest.RandomForestt as TTRF
    import Scripts.traintest.SupportVectorMachinett as TTSVM
    import Scripts.traintest.NeuralNetworktt as TTNN
    
        
    #%%Run Gridsearch
    
    #Logistic Regression
    # GLR.run_LR_gridsearch(shading=True)
    # GLR.run_LR_gridsearch(shading=False)
        
    #Decision Tree
    # DT.overfit_DT(shading=True,runs=100)
    # DT.overfit_DT(shading=False,runs=100)
    # DT.run_DT_gridsearch(shading=True)
    # DT.run_DT_gridsearch(shading=False)
    
    #Random Forest    
    # RF.run_RF_gridsearch(shading=True)
    # RF.run_RF_gridsearch(shading=False)
    
    #Support Vector Machine
    # SVM.run_SVM_gridsearch(shading=True)
    # SVM.run_SVM_gridsearch(shading=False)
    
    #Neural Network
    # NN.run_NN_gridsearch(shading=True)
    # NN.run_NN_gridsearch(shading=False)


    #%%Run Test-split repeaditly
    
    #Logistic Regression
    TTLR.run_LR_traintest(shading=True,num_iterations=100)
    TTLR.run_LR_traintest(shading=False,num_iterations=100)
    
    #Decsion Tree
    TTDT.run_DT_traintest(shading=True,num_iterations=100)
    TTDT.run_DT_traintest(shading=False,num_iterations=100)
    
    # #Random Forest
    TTRF.run_RF_traintest(shading=True,num_iterations=100)
    TTRF.run_RF_traintest(shading=False,num_iterations=100)
    
    # #Support Vector Machine
    TTSVM.run_SVM_traintest(shading=True,num_iterations=100)
    TTSVM.run_SVM_traintest(shading=False,num_iterations=100)
    
    #Neural Network
    TTNN.run_NN_traintest(shading=True,num_iterations=100)
    TTNN.run_NN_traintest(shading=False,num_iterations=100)
    
        
    #%%DOT export 
    #if render with graphviz failed (for DT and RF)
    # UT.plot_from_dot_file("Overfitted_tree", shading=True, to_file="DT",
    #                       output_filename="Decision_tree",vector_export=True
    #                       ,png_export=False)
    