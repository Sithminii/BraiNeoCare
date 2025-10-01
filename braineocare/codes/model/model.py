import torch
import torch.nn as nn
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SE_Attention(nn.Module):

    '''Squeeze and excitation attention'''

    def __init__(self, in_channels, reduction=16):
        super(SE_Attention, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        attn = self.fc(x)
        return x * attn
    


class MFCCModel(nn.Module):

    ''' Seizure prediction model architecture '''

    def __init__(self, in_channels:int=19, dropout:float=0.1, reduction:int=16):
        super(MFCCModel, self).__init__()

        self.in_chs = in_channels
        self.dropout = dropout
        self.reduction = reduction

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_chs, out_channels=32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_chs, out_channels=32, kernel_size=(5,5), padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU()
        )  
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,1)),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU()
        )
        self.attention = SE_Attention(in_channels=128, reduction=self.reduction)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128*1*1,1)
        
    def forward(self, x):  
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)

        x = x1 + x2

        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.attention(x)
        x_out = self.classifier(self.flatten(x))

        return x_out
    
