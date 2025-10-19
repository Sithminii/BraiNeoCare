import os
import numpy as np

from src.dataset.generate_dataset import read_and_extract_data
from conf.conf import get_config


conf = get_config()

# Cutoff frequencies of the denoising filters
filter_cfg = conf['data_preprocessing']['filter']
filter_data = {'bandpass': filter_cfg['bandpass'],
               'notch': filter_cfg['notch']}

# Important durations in seconds
durations_cfg = conf['data_preprocessing']
duration_data = {'boundary': 2*60,
                 'segment_size': durations_cfg['segment_duration'],
                 'preictal_duration': durations_cfg['preictal_duration'],
                 'postictal_duration': durations_cfg['postictal_duration'],
                 'int_end_to_ict_duration': durations_cfg['int_end_to_ict_duration']}

# Define MFCC specifications
mfcc_cfg = conf['data_preprocessing']['mfcc']
mfcc_specs = {'n_mfcc': mfcc_cfg['n_mfcc'],
              'n_mels': mfcc_cfg['n_mels'],
              'hop_length': mfcc_cfg['hop_length'],
              'f_min': mfcc_cfg['f_min'],
              'f_max': mfcc_cfg['f_max']}

# Define file and folder paths
data_path = conf['paths']['zenodo_dataset_path']
dest_path = conf['paths']['dataset_path']
annotations_path = os.path.join(data_path, 'annotations_2017.mat')

# Generate the interictal and preictal dataset
read_and_extract_data(data_folder=data_path,
                      dest_path=dest_path,
                      annotation_file=annotations_path,
                      filter_data=filter_data,
                      duration_data=duration_data,
                      mfcc_specs=mfcc_specs)