import sys
import os
import torch

from src.explainable_AI.XAI_utils import predict_and_visualize_shap
from conf.conf import get_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    conf = get_config()
    
    data_dir = os.path.join(conf['paths']['test_path'], 'neon_11')
    dest_dir = conf['paths']['output_path']

    predict_and_visualize_shap(data_dir, dest_dir, device)


if __name__ == '__main__':
    main() 