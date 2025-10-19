import os
import numpy as np
import torch
import torch.nn as nn
import warnings
from model_tune import MFCCModel
import data_tune
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import joblib

from conf.conf import get_config

warnings.simplefilter("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def summarize(metric):
    mean = np.mean(cv_res[f'mean_test_{metric}'])
    std = np.mean(cv_res[f'std_test_{metric}'])
    return mean, std


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


class HyperTune(NeuralNetClassifier):
    def __init__(self, *args, threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold= threshold
    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= self.threshold).astype(np.int64)


conf = get_config()

# Paths to the dataset and output directories
dataset_path = conf['paths']['dataset_path']
neon = 0

# Number of channels (EEG+ECG)
num_channels = 23

dropout = 0.1
model = MFCCModel(in_channels=num_channels, dropout=dropout)
model.to(device)

data_dict = data_tune.read_data(dataset_path, neon)
X_train, y_train, groups = data_dict['data'], data_dict['labels'], data_dict['groups']

print("dataset size = ", len(y_train))


tuning_net = HyperTune(
    model,
    max_epochs=150,
    iterator_train__shuffle = True,
    optimizer=torch.optim.Adam,
    criterion = nn.BCEWithLogitsLoss,
    threshold=0.5,

    iterator_train__batch_size=64,
    iterator_valid__batch_size=128,
    callbacks=[
        EpochScoring('average_precision', name='val_ap', on_train=False),
        EpochScoring('roc_auc', name='val_roc', on_train=False),
        EpochScoring('f1', name='val_f1', on_train=False)
        ],
    device = device
)


param_grid = {
    'lr': [0.0001, 0.0002, 0.0004, 0.0005, 1e-2, 2e-2],
    'optimizer__weight_decay': [1e-6, 5e-5, 1e-4, 1e-3],
    'module__dropout': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.3],
    'criterion__pos_weight': [torch.tensor([0.45]), torch.tensor([0.46]), torch.tensor([0.47]), torch.tensor([0.48]), torch.tensor([0.49]),torch.tensor([0.5]), torch.tensor([0.51]), torch.tensor([0.52]), torch.tensor([0.53]), torch.tensor([0.54]), torch.tensor([0.55]), torch.tensor([0.56]), torch.tensor([1.0]), torch.tensor([1.2])],
    'iterator_train__batch_size': [64, 128, 256],
    'module__in_channels':[num_channels],
    'module__reduction':[2, 4, 8, 16, 32],
    'threshold': [0.5]
}

specificity_scorer = make_scorer(specificity_score)

gs = RandomizedSearchCV(tuning_net,
                        param_grid,
                        cv=5,
                        scoring = {
                            'accuracy': 'accuracy',
                            'sensitivity': 'recall',
                            'specificity': specificity_scorer,
                            'f1': 'f1',
                            'bal_acc': 'balanced_accuracy',
                            'ap': 'average_precision',
                            'roc': 'roc_auc',
                        },
                        refit='ap',
                        n_jobs=1,
                        verbose=2,
                        random_state=42)
gs.fit(X_train, y_train)


cv_res = gs.cv_results_


acc_mean, acc_std   = summarize("accuracy")
sens_mean, sens_std = summarize("sensitivity")
spec_mean, spec_std = summarize("specificity")
f1_mean, f1_std     = summarize("f1")

print("Cross-validation results (5 folds):")
print(f"Accuracy:    {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Sensitivity: {sens_mean:.4f} ± {sens_std:.4f}")
print(f"Specificity: {spec_mean:.4f} ± {spec_std:.4f}")
print(f"F1-score:    {f1_mean:.4f} ± {f1_std:.4f}")


print("Best params:", gs.best_params_)
print("Best score:", gs.best_score_)


# best model from randomized CV
best_net = gs.best_estimator_

# Save weights
torch.save(best_net.module_.state_dict(), f"/home/braineocare/results/best_model_weights.pth")

# Save full skorch object
joblib.dump(best_net, f"/home/braineocare/results/best_skorch_model.pkl")
