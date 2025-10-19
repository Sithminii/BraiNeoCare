import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import warnings

from src.utility.utils import prob_to_binary, compute_metrics

warnings.simplefilter("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model,
          train_dataloader,
          val_dataloader,
          optimizer,
          criterion,
          scheduler,
          train_loss:list,
          val_loss:list,
          train_acc_arr:list,
          val_acc_arr:list,
          epoch:int):
    
    '''
    A single iteration of model training and validation
    '''

    # Model_training
    epoch_loss = 0
    correct=0
    total_preds = 0

    model.train()
    for train_x, label in train_dataloader:
        # Pass data to device
        train_x = train_x.to(device)
        label = label.to(device, dtype=torch.float).reshape(-1,1)

        # Set optimizer gradients to zero
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(train_x)
        # Backward pass
        loss = criterion(y_pred, label)
        loss.backward()
        epoch_loss += loss.item()

        # Update the weights
        optimizer.step()
        
        # Convert predicted logits probabilities and then to binary outputs 
        with torch.no_grad():
            y_pred = F.sigmoid(y_pred)
            y_pred_bin = prob_to_binary(y_pred.cpu().numpy())
            total_preds += len(y_pred_bin)
            correct += (y_pred_bin==label.cpu().numpy()).sum()

    epoch_loss /= len(train_dataloader)
    train_loss.append(epoch_loss)

    train_accuracy = correct/total_preds
    train_acc_arr.append(train_accuracy)
    

    # Model_validation
    epoch_loss = 0
    total_preds = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for val_x, label in val_dataloader:
            # Pass data to device
            val_x = val_x.to(device)
            label = label.to(device, dtype=torch.float).reshape(-1,1)

            # Forward pass
            y_pred = model(val_x)

            # Compute validation loss
            loss = criterion(y_pred, label)
            epoch_loss += loss.item()

            # Convert predicted logits to probabilities and then to binary outputs
            y_pred = F.sigmoid(y_pred)
            y_pred_bin = prob_to_binary(y_pred.cpu().numpy())
            total_preds += len(y_pred_bin)
            correct += (y_pred_bin == label.cpu().numpy()).sum()

        epoch_loss /= len(val_dataloader)
        val_loss.append(epoch_loss)
        
        scheduler.step(epoch_loss)
        
        val_accuracy = correct/total_preds
        val_acc_arr.append(val_accuracy)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f'epoch {epoch+1} seiz_class: train={train_loss[-1]:.12f} , val={val_loss[-1]:.12f} & lr={current_lr}')
    


def evaluate_model(model, dataloader, validation=False):
    '''
    Evaluate the performance of the trained model on a validation/ test dataset
    '''
    conf_mat = 0

    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            # Pass data to device
            data = data.to(device)
            label = label.reshape(-1,1)
            
            # Forward pass
            y_pred = model(data)

            # Convert to probabilities and then to binary outputs
            y_pred = F.sigmoid(y_pred)
            y_pred = prob_to_binary(y_pred.cpu().numpy())

            # Compute the confusion matrix
            conf_mat += confusion_matrix(label, y_pred, labels=[0,1]).ravel()

    # Compute performance metrics
    acc, sen, spec, f1_score = compute_metrics(conf_mat)

    if validation:
        print(f'\tValid : Acc= {acc}; Sen= {sen}; Spec= {spec}; F1-score= {f1_score}')
    else:
        print(f'\tUnseen: Acc= {acc}; Sen= {sen}; Spec= {spec}; F1-score= {f1_score}')

    metric_dict = {'acc':acc, 'sen':sen, 'spec':spec, 'f1_score':f1_score}
    
    return metric_dict