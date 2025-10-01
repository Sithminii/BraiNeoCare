import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings


from braineocare.codes.utility.data import read_data, CreateDataset
from braineocare.codes.utility.plot_results import loss_and_accuracy, performance
from braineocare.codes.model.model import MFCCModel
from braineocare.codes.model.model_utils import train_model, evaluate_model


warnings.simplefilter("ignore", category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths to the dataset and output directories
dataset_path = '/home/braineocare/datasets/zenodo_dataset'
output_path = '/home/braineocare/results/zenodo_kfold/checkpts'
fold_dest = '/home/braineocare/results/zenodo_kfold/fold_data'

# Generate folds and overwrite exisiting
# Set to 'False' to use previous folds generated
write_folds = True

# Number of channels (EEG+ECG)
num_channels = 23

# Model hyperparameters
batch_size = 256
epochs = 300
learning_rate = 0.0004
weight_decay = 0.005
n_folds = 10
dropout = 0.3
positive_weight = torch.tensor([0.52], dtype=torch.float).to(device)
model_reduction = 8


print('\n','-'*100)
print(f'Performing {n_folds}-fold cross-validation...')
print(f'learning rate: {learning_rate}')
print(f'batch_size: {batch_size}')
print(f'epochs: {epochs}')
print(f'weight decay: {weight_decay}')
print(f'dropout: {dropout}')
print('-'*100,'\n')

# Arrays to store fold-wise results
fold_acc = torch.zeros(n_folds)
fold_sen = torch.zeros(n_folds)
fold_spec = torch.zeros(n_folds)
fold_f1score = torch.zeros(n_folds)
fold_trainloss = torch.zeros(n_folds)
fold_valloss = torch.zeros(n_folds)


for fold in range(n_folds):

    print(f'Training fold_{fold+1}...')
    
    # Separate train and test data
    train_f = []

    if write_folds:
        if fold == 0: # Generates fold-dataset and saves each fold
            all_folds = read_data(dataset_path, mode = 'kfold', n_folds=n_folds)
            for k in range(n_folds):
                np.save(os.path.join(fold_dest,f'fold_list_{k}.npy'), np.array(all_folds[k]))
                if k!=fold:
                    train_f.extend(all_folds[k])
                else:
                    test_f = all_folds[k].copy()  
        else: # Extract train and test data from the saved folds
            for k in range(n_folds):
                if k!=fold:
                    files=list(np.load(os.path.join(fold_dest,f'fold_list_{k}.npy')))
                    train_f.extend(files)
                else:
                    test_f=list(np.load(os.path.join(fold_dest,f'fold_list_{k}.npy')))
    else:
        for k in range(n_folds):
            fold_path = os.path.join(fold_dest,f'fold_list_{k}.npy')
            data_paths = list(np.load(fold_path))
            if k != fold:
                train_f.extend(data_paths)
            else:
                test_f = data_paths.copy()

    print('Train data = ', len(train_f))
    print('Test data = ', len(test_f))

    # Create a folder to store results for the current fold
    cur_out_path = os.path.join(output_path, f'fold_{fold+1}')
    os.makedirs(cur_out_path, exist_ok=True)

    # Generate train and validation dataloaders
    trainset = CreateDataset(maindir=dataset_path, f_list=train_f)
    valset = CreateDataset(maindir=dataset_path, f_list=test_f) 
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=64, shuffle=False)
    
    # Initialize the model
    model = MFCCModel(in_channels=num_channels, dropout=dropout, reduction=model_reduction)
    model.to(device)
    
    # Define optimizer, learning rate scheduler and the criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, factor = 0.98, min_lr=1e-7)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    
    # Lists to store training and validation results
    train_loss, val_loss, train_acc_arr, val_acc_arr = [],[],[],[]
    # Lists to store validation performance scores
    val_acc, val_sen, val_spec, val_f1 = [],[],[],[]        
    
    for epoch in range(epochs):
        train_model(model,
                    train_dataloader, val_dataloader,
                    optimizer, criterion, scheduler,
                    train_loss, val_loss,
                    train_acc_arr, val_acc_arr,
                    epoch)
    
        if (epoch+1)%5 == 0:
            val_metric_dict = evaluate_model(model, val_dataloader, validation=True)
            
            val_acc.append(val_metric_dict['acc'])
            val_sen.append(val_metric_dict['sen'])
            val_spec.append(val_metric_dict['spec'])
            val_f1.append(val_metric_dict['f1_score'])
            
            checkpoint = {
                'epoch':epoch+1,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':train_loss[-1],
                'val_loss':val_loss[-1],
                'val_metrics': val_metric_dict
            }
            torch.save(checkpoint, os.path.join(cur_out_path,f'checkpoint_epoch_{epoch+1}.pth'))
    
    fold_acc[fold] = val_metric_dict['acc']
    fold_sen[fold] = val_metric_dict['sen']
    fold_spec[fold] = val_metric_dict['spec']
    fold_f1score[fold] = val_metric_dict['f1_score']
    fold_trainloss[fold] = train_loss[-1]
    fold_valloss[fold] = val_loss[-1]

            
    # Plot results
    loss_and_accuracy(train_loss, val_loss, train_acc_arr, val_acc_arr, cur_out_path)
    performance(val_acc, val_sen, val_spec, val_f1, epochs, cur_out_path)     
    
    torch.cuda.empty_cache()
