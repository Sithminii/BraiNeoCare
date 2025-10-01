import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import warnings

from braineocare.codes.utility.utils import get_ID

warnings.simplefilter("ignore", category=FutureWarning)


class CreateDataset(Dataset):
    def __init__(self, maindir, f_list):
        self.datalist = sorted(f_list, key=get_ID) #File names
        self.maindir = maindir # Dataset folder path
    
    def __len__(self):
        return len(self.datalist)
        
    def __getitem__(self, index):
        # Read data file
        mfcc = np.load(os.path.join(self.maindir, self.datalist[index]))
        mfcc = torch.tensor(mfcc, dtype=torch.float)
        
        # Generate the label; preictal-1 & interictal-0
        if self.datalist[index].split('.')[0][-3:] == 'int':
            label = 0
        elif self.datalist[index].split('.')[0][-3:] == 'pre':
            label = 1
        
        return mfcc, label
    


def read_data(maindir, mode='kfold', n_folds=10, test_patient=None):

    '''
    Reads a dataset and returns train and/or test data according to the validation mode specified

    inputs:
        maindir - path to the dataset
        mode - type of cross validation; 'lopo' or 'kfold', Default 'kfold'
                    'kfold': k-fold cross-validation
                    'lopo' : leave-one-patient-out cross validation
        n_folds - number of folds if mode is 'kfold'; Default 10
        test_patient - name of the folder of the testing (leaving out) patient in 'lopo'

    output:
        if mode = 'kfold' - list of k-folds of the dataset
        if mode = 'lopo' - train and test file lists
    '''

    allfolders = sorted(os.listdir(maindir), key=get_ID)

    train_f = []
    
    if mode=='kfold':
        # Iterate over each patient and add files to the train_f list
        for folder in allfolders:
            folder_path = os.path.join(maindir,folder)
            files = sorted([os.path.join(folder,file) for file in os.listdir(folder_path)], key=get_ID)         
            train_f.extend(files)
        
        # Shuffle and partition the list of data files into n_folds
        random.shuffle(train_f)
        folds = list(np.array_split(np.array(train_f), n_folds))

        return folds


    elif mode=='lopo':
        assert test_patient is not None

        for folder in allfolders:
            if folder != test_patient:
                folder_path = os.path.join(maindir,folder)
                files = sorted([os.path.join(folder,file) for file in os.listdir(folder_path)], key=get_ID)
                train_f.extend(files)
            else:
                print("Extracting test data")
                test_folder = os.path.join(maindir,test_patient)
                test_f = sorted([os.path.join(test_patient,file) for file in os.listdir(test_folder)], key=get_ID)
        
        print(f'No. of samples: train = {len(train_f)} & test = {len(test_f)}')
        data = {'train':train_f, 'test':test_f}

        return data
    
    else:
        raise ValueError(f'Provided cross-validation method is unknown. Expected either \'kfold\' or \'lopo\', got {mode}')
    
