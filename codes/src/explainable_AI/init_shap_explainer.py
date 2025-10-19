import os
import torch

from XAI_utils import generate_shap_explainer
from conf.conf import get_config

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    conf = get_config()

    data_dir = conf['paths']['train_path']
    checkpt_path = conf['paths']['weights_path']

    checkpt = torch.load(checkpt_path, weights_only=False, map_location=device)
    weights = checkpt['model_state_dict']

    generate_shap_explainer(in_channels=19,
                            reduction=8,
                            model_state_dict=weights,
                            train_data_dir=data_dir,
                            device=device)
    
if __name__ == '__main__':
    main()

