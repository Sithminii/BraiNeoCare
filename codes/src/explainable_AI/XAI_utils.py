import shap
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from src.utility.utils import get_ID
from src.utility.data import CreateDataset
from src.model.model import MFCCModel, SigmoidWrappedModel
from conf.conf import get_config

conf = get_config()

def save_shap_explainer(explainer):
    with open(conf['path']['explainer_path'], 'wb') as f:
        pickle.dump(explainer, f)


def load_shap_explainer():
    with open(conf['path']['explainer_path'], 'rb') as f:
        explainer = pickle.load(f)
    return explainer


def generate_shap_explainer(in_channels, reduction, model_state_dict, train_data_dir, device='cpu'):
    # Initialize the model
    model = MFCCModel(in_channels=in_channels, reduction=reduction)
    model.load_state_dict(model_state_dict)
    model.eval()

    model = SigmoidWrappedModel(model).to(device)
    torch.save(model, conf['path']['trained_model_path'])

    # Initialize train data
    all_files = []
    all_folders = os.listdir(train_data_dir)
    
    for folder in all_folders:
        folder_files = os.listdir(os.path.join(train_data_dir, folder))
        files = [os.path.join(folder,f) for f in folder_files if f.endswith('.npy')]
        all_files.extend(files)

    train_dataset = CreateDataset(train_data_dir, all_files)
    bg_dataloader = DataLoader(train_dataset, batch_size=len(all_files), shuffle=False)

    for x,_ in bg_dataloader:
        x_train = x.to(device)
    
    explainer = shap.GradientExplainer(model, x_train)
    save_shap_explainer(explainer)



def generate_shap_scalp_plot(shap_array, idx, dest_path):

    img_path = conf['paths']['electrode_img__path']
    
    channel_locs = {
        'Fp1-T3':((-0.3, 0.65), (-0.75, 0.1)),
        'T3-O1':((-0.3, -0.65), (-0.75, -0.1)),
        'Fp1-C3':((-0.24, 0.65), (-0.39, 0.1)),
        'C3-O1':((-0.39, -0.1), (-0.24, -0.65)),
        'Fp2-C4':((0.24, 0.65), (0.39, 0.1)),
        'C4-O2': ((0.39, -0.1), (0.24, -0.65)),
        'Fp2-T4': ((0.3, 0.65), (0.75, 0.1)),
        'T4-O2': ((0.3, -0.65), (0.75, -0.1)),
        'T3-C3': ((-0.68, 0.0), (-0.49, 0.0)),
        'C3-Cz': ((-0.29, 0.0), (-0.1, 0.0)),
        'Cz-C4': ((0.1, 0.0), (0.29, 0.0)),
        'C4-T4': ((0.49, 0.0), (0.68, 0.0)),
        'Fp1-Cz': ((-0.20, 0.65), (-0.08, 0.06)),
        'Cz-O1': ((-0.20, -0.65), (-0.08, -0.06)),
        'Fp2-Cz': ((0.20, 0.65), (0.08, 0.06)),
        'Cz-O2': ((0.20, -0.65), (0.08, -0.06)),
        'Fp1-O2': ((-0.2, 0.65), (0.2, -0.65)),
        'Fp2-O1': ((0.2, 0.65), (-0.2, -0.65))
    }

    # 18 bipolar channels
    channels = [
        'Fp1-T3', 'T3-O1', 'Fp1-C3', 'C3-O1', 'Fp2-C4', 'C4-O2',
        'Fp2-T4', 'T4-O2', 'T3-C3', 'C3-Cz', 'Cz-C4', 'C4-T4',
        'Fp1-Cz', 'Cz-O1', 'Fp2-Cz', 'Cz-O2', 'Fp1-O2', 'Fp2-O1'
    ]

    # Colormap
    cmap = cm.Reds
    norm = plt.Normalize(vmin=np.min(shap_array), vmax=np.max(shap_array))

    fig, ax = plt.subplots(figsize=(6,6))

    # Add scalp background image
    img = plt.imread(img_path)  # your scalp image
    ax.imshow(img, extent=[-1.2, 1.2, -1.1077, 1.1077], aspect='auto')

    # Plot channels as lines colored by SHAP value
    for ch, shap in zip(channels, shap_array):
        (x1,y1), (x2,y2) = channel_locs[ch]
        ax.plot([x1, x2], [y1, y2], color=cmap(norm(shap)), linewidth=3)

    # Colorbar for SHAP values
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.035, pad=0.04)
    cbar.set_label("SHAP Importance", fontsize=12)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    cur_time = str(datetime.datetime.now())
    plt.savefig(os.path.join(dest_path,f'scalp_plot_{idx}_{cur_time}.png'))



def predict_and_visualize_shap(data_dir, dest_dir, device='cpu'):
    test_files = sorted(os.listdir(data_dir), key=get_ID)
    test_dataset = CreateDataset(data_dir, test_files, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    model_path = conf['paths']['trained_model_path']
    model = torch.load(model_path, map_location=device, weights_only=False)

    model.eval()
    with torch.no_grad():
        for x in test_dataloader:
            x_test = x.to(device)
            y_test = model(x_test)

    output_df = pd.DataFrame({
        'data_sample': np.array(test_files),
        'is_preictal': y_test.cpu().numpy().squeeze()
        })
    output_df.to_csv(os.path.join(dest_dir, 'prediction_results.csv'), index=False)

    # Load shap explainer and base probability
    explainer = load_shap_explainer()

    # Get shap values
    shap_values_raw = explainer.shap_values(x_test)
    shap_values_lst = shap_values_raw.squeeze(-1)

    # Create a folder to store scalp plots
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scalp_plot_dest = os.path.join(dest_dir, f'scalp_plots_{cur_time}')
    os.makedirs(scalp_plot_dest, exist_ok=True)
    print(f'Saving data to path "{scalp_plot_dest}"')

    for idx in range(len(test_dataset)):
        # Average with last two shap matrices if available
        if idx < 2:
            avg_shap = shap_values_lst[idx]
        else:
            shap_window = shap_values_lst[idx-2 : idx+1]
            avg_shap = shap_window.mean(axis=0)

        # Get per channel shap values
        shap_array = avg_shap.mean(axis=(1,2))/np.std(avg_shap, axis=(1,2))
        shap_array[shap_array < 0] = 0

        # Remove ECG channel
        shap_array = shap_array[:-1]

        # Ensure shap array is between [0,1]
        shap_array = (shap_array - np.min(shap_array))/(np.max(shap_array) - np.min(shap_array))

        scalp_plot_id = test_files[idx]
        generate_shap_scalp_plot(shap_array, scalp_plot_id, scalp_plot_dest)



