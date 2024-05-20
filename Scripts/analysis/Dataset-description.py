# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:22:18 2024

@author: Kenny
"""

from Scripts.util import (
    get_data,
    filter_data,
    generate_table,
    train_test_split_data,
    perform_grid_search,
    get_performance_metrics,
    get_confusion_matrix,
    plot_confusion_matrix,
    write_output_to_csv,
    save_object_to_file
)

def dataset_description():
    pass

raw_data=get_data()

#case Shading = True
shading=True
data_shading=filter_data(raw_data,filter_value=100,shading=shading)
print('Fault distribution WITH SHADING')
generate_table(raw_data,data_shading,"Raw","Filtered")

#case Shading = False
shading=False
data_woShading=filter_data(raw_data,filter_value=100,shading=shading)
print('Fault distribution WITHOUT SHADING')
generate_table(raw_data,data_woShading,"Raw","Filtered")