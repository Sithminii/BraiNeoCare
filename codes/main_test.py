import sys
import os
import torch

from src.explainable_AI.XAI_utils import test_model
from conf.conf import get_config


warnings.simplefilter("ignore", category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    conf = get_config()
    
    data_dir = os.path.join(conf['paths']['test_path'], 'neon_11')
    dest_dir = conf['paths']['output_path']

    test_model(data_dir, dest_dir, device)


if __name__ == '__main__':
    main() 