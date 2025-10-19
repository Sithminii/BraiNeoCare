import os
import numpy as np
import torch
import torch.nn as nn
import warnings
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
import joblib

from src.model.model import MFCCModel
from src.utility.utils import get_ID
from conf.conf import get_config

warnings.simplefilter("ignore", category=FutureWarning)

conf = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to the dataset and output directories
dataset_path = conf['paths']['dataset_path']
neon = 66

checkpoint_maindir = conf['paths']['lopo_path']
checkpoint_pth = os.path.join(checkpoint_maindir, f'/neon_{neon}/neon_{neon}_run1/checkpoint_epoch_15.pth')

# Number of test samples
N = 12

# Generate the dataset
full_path = os.path.join(dataset_path, f"neon_{neon}")
int_files = [f for f in os.listdir(full_path) if f.split('.')[0][-3:] == 'int']
pre_files = sorted([f for f in os.listdir(full_path) if f.split('.')[0][-3:] == 'pre'], key=get_ID)

files = []
files.extend(pre_files)
files.extend(int_files)

x_train = []
y_train = []
for f in files:
    path = os.path.join(full_path, f)
    x_train.append(np.load(path))
    if f.split('.')[0][-3:]=='int':
        y_train.append(0)
    elif f.split('.')[0][-3:]=='pre':
        y_train.append(1)
    else:
        print("Unknown class!")

x_train = np.array(x_train)
y_train = np.array(y_train, dtype=np.float32).reshape(-1,1)

groups = np.zeros(len(files))
groups[:N] = -1
groups[len(pre_files): len(pre_files)+N] = -1

num_channels = 19 # Number of channels (EEG+ECG)
dropout = 0.1

checkpt_data = torch.load(checkpoint_pth, weights_only=False, map_location=device)

model = MFCCModel(in_channels=num_channels, reduction=8, dropout=dropout)
model.load_state_dict(checkpt_data['model_state_dict'])
model.to(device)

print("dataset size = ", len(y_train))


class HyperTune(NeuralNetClassifier):
    def __init__(self, *args, threshold=0.5, pretrained_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.pretrained_params = pretrained_params  # path to pretrained weights

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= self.threshold).astype(np.int64)

    def initialize_module(self):
        # Run normal initialization
        super().initialize_module()
        # Load pretrained weights if provided
        if self.pretrained_params is not None:
            self.module_.load_state_dict(self.pretrained_params)
        return self


tuning_net = HyperTune(
    module=model,
    max_epochs=150,
    iterator_train__shuffle=True,
    optimizer=torch.optim.Adam,
    criterion=nn.BCEWithLogitsLoss,
    threshold=0.5,
    pretrained_params=checkpt_data['model_state_dict'],
    iterator_train__batch_size=64,
    iterator_valid__batch_size=128,
    callbacks=[
        EpochScoring('average_precision', name='val_ap', on_train=False),
        EpochScoring('roc_auc', name='val_roc', on_train=False),
        EpochScoring('f1', name='val_f1', on_train=False)
    ],
    device=device,
)


param_grid = {
    'lr': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1e-3, 1e-2],
    'optimizer__weight_decay': [1e-10, 1e-8, 1e-6, 5e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    'module__dropout': [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.2, 0.3],
    'criterion__pos_weight': [torch.tensor([0.1]), torch.tensor([0.15]),torch.tensor([0.2]), torch.tensor([0.25]),torch.tensor([0.3]), torch.tensor([0.35]), torch.tensor([0.45]), torch.tensor([0.46]), torch.tensor([0.47]), torch.tensor([0.48]), torch.tensor([0.49]),torch.tensor([0.5]), torch.tensor([0.51]), torch.tensor([0.52]), torch.tensor([0.53]), torch.tensor([0.54]), torch.tensor([0.55]), torch.tensor([0.56]), torch.tensor([1.0]), torch.tensor([1.2])],
    'iterator_train__batch_size': [32, 64, 128],
    'module__in_channels':[num_channels],
    'module__reduction':[8],
    'threshold': [0.5]
}


ps = PredefinedSplit(groups)

train_size = 0
test_size = 0
for fold_idx, (train_idx, test_idx) in enumerate(ps.split()):
    print(f"Fold {fold_idx}:")
    print("Train size:", len(train_idx))
    print("Test size:", len(test_idx))

    train_size += len(train_idx)
    test_size += len(test_idx)
    
    # Ensure no data leak between train and test data
    print(set(train_idx).intersection(set(test_idx)))


gs = RandomizedSearchCV(
    tuning_net,
    param_distributions=param_grid,  # should be param_distributions, not param_grid
    cv=ps,
    scoring={
        'ap': 'average_precision',
        'roc': 'roc_auc',
        'f1': 'f1',
        'bal_acc': 'balanced_accuracy'
    },
    refit='ap',
    n_jobs=1,
    n_iter=10,
    random_state=42,
    verbose=2,
)

gs.fit(x_train, y_train)

print("Best params:", gs.best_params_)
print("Best score:", gs.best_score_)


x_test = x_train[groups==0]
y_test = y_train[groups==0]

print(y_test.shape)

# best model from randomized CV
best_net = gs.best_estimator_


joblib.dump(best_net, f"best_skorch_model_{neon}.pkl")

best_net = joblib.load(f"best_skorch_model_{neon}.pkl")


y_pred = best_net.predict(x_test)

# Standard metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Sensitivity
sensitivity = tp / (tp + fn)

# Specificity
specificity = tn / (tn + fp)

print(f"Accuracy:    {acc:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")