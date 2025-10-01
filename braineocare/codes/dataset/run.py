import os
import numpy as np

from generate_dataset import read_and_extract_data


# Cutoff frequencies of the denoising filters
filter_data = {'bandpass':[0.1,70],
               'notch':50}

# Important durations in seconds
duration_data = {'boundary':2*60,
                 'segment_size':5,
                 'preictal_duration':30*60,
                 'postictal_duration':30*60,
                 'int_end_to_ict_duration':60*60}

# Define MFCC specifications
mfcc_specs = {'n_mfcc':20, 'n_mels':20, 'hop_length':128, 'f_min':0.1, 'f_max':100}

# Define file and folder paths
data_path = 'Zenodo_dataset'
dest_path = 'Model_Zenodo_Dataset'
annotations_path = 'Zenodo_dataset\\annotations_2017.mat'

# Generate the interictal and preictal dataset
read_and_extract_data(data_folder=data_path,
                      dest_path=dest_path,
                      annotation_file=annotations_path,
                      filter_data=filter_data,
                      duration_data=duration_data,
                      mfcc_specs=mfcc_specs)