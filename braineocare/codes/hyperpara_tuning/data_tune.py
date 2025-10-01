import os
import numpy as np
import warnings
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from braineocare.codes.utility.utils import get_ID

warnings.simplefilter("ignore", category=FutureWarning)    



def read_data(data_dir, neon=None): 

    allfolders = sorted(os.listdir(data_dir), key=get_ID)

    X_train = []
    y_train = []
    groups = []

    for folder in allfolders:
        print(f"Reading {folder}")

        group_id = get_ID(folder)
        print('Group: ', group_id)

        folder_path = os.path.join(data_dir,folder)
        files = sorted([os.path.join(folder_path,file) for file in os.listdir(folder_path)], key=get_ID)

        for file in files:
            X_train.append(np.load(file))

            if group_id == neon:
                groups.append(0)
            else:
                groups.append(-1)

            if file.split('.')[0][-3:]=='int':
                y_train.append(0)
            elif file.split('.')[0][-3:]=='pre':
                y_train.append(1)
            else:
                print('Unknown label found!')
                break

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32).reshape(-1,1)
    groups = np.array(groups, dtype=int)

    return {'data':X_train, 'labels':y_train, 'groups': groups}


def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def combined_scorer(estimator, X, y):
    y_pred = estimator.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    sens = sensitivity_score(y, y_pred)
    spec = specificity_score(y, y_pred)
    
    print(f"Val Acc: {acc:.3f}, F1: {f1:.3f}, Sens: {sens:.3f}, Spec: {spec:.3f}")
    
    return f1

