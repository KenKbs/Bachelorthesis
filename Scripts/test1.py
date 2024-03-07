# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:52:00 2024

@author: Kenny
"""

#%% Imports:
from gridsearch import LogisticRegression as LRGS
from util import load_object_from_file, get_data, filter_data,generate_table


raw_data=get_data()

#Filter out low irrediance values
data=filter_data(raw_data,filter_value=100)
shaded_data=filter_data(raw_data,filter_value=100,shading=False)

#Print fault distribution before and after fitering
generate_table(raw_data,data,"raw","filtered") 

generate_table(raw_data,shaded_data,"raw","shaded")

generate_table(data,shaded_data,"filtered","shaded")


# if __name__=="__main__":
#     LRGS.run_LR_gridsearch(shading=True) #Runs Gridsearch with shading class inlcuded
#     LRGS.run_LR_gridsearch(shading=False) #Runs Gridsearch with shading class excluded
    
#     #Test load objects
#     best_model_LR=load_object_from_file(file_name=r"Best_Model_LR.pk1",
#                                         to_file="LR",shading=True)
    
#     cm_LR=load_object_from_file(file_name="Grid-search_CM_LR.pk1",
#                                         to_file="LR",shading=True)
    
#     report_LR=load_object_from_file(file_name="Grid-search_report_LR.pk1",
#                                     file_path=r"C:\Users\Kenny\Dropbox\Education\Uni\FU-Berlin\Inhalte\9. Semester\Bachelorarbeit\Programmierung\GitHub-Repo\Bachelorthesis\Results\LR\Shading"
                                    # )
    

