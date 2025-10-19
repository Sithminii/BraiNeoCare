# Explainable Deep Learning for Neonatal Seizure Prediction with Reduced Montage

## Problem Background

- **High susceptibility and impact:**  
  Newborns during the first 28 days of life (the neonatal period) are especially vulnerable to seizures because their central nervous systems are immature. Early detection and intervention are critical to reduce the risk of secondary brain injury and long-term neurodevelopmental impairments.

- **Challenges in clinical detection:**  
  Seizures in neonates often present with very subtle or ambiguous signs that can be mistaken for normal infant behavior, making reliable detection based solely on bedside clinical observation difficult in the NICU.

- **Limitations of current monitoring methods:**  
  Continuous video-EEG (video-cEEG) is the clinical gold standard for neonatal seizure detection, but real-time monitoring requires continuous expert review and is resource-intensive, limiting its availability and practicality in many clinical settings.

## Our approach

### Proposed Model Architecture

<img width="1326" height="595" alt="image" src="https://github.com/user-attachments/assets/5c254abf-7025-4cf3-8b90-6011cf789948" />

### Create a Conda ENvironment

If you haven't installed Conda on your PC, please install it by referring to the following link.
[https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Once you have successfully installed Conda on your PC, you can start by creating a Conda environment for the project.
```bash
conda activate myenv
```

### Clone the GitHub repository

Next, clone the GitHub repository by running the following command in your terminal after navigating to a preferred location. Or else you can download the zip from the GitHub repo.

```bash
git clone https://github.com/Sithminii/BraiNeoCare.git
```
Then get into the cloned folder.
```bash
cd BraiNeoCare
```

### Install required libraries
Inside the BraiNeoCare folder, run the following command to install all the required libraries.
```bash
pip install -r requirements.txt
```
### Dataset
To run the files, you need to download the publicly available Zenodo Neonatal EEG dataset published by Helsinki University. You can find the dataset [here](https://zenodo.org/records/4940267). Please make sure to download version 4 of the dataset. It is recommended to create a folder named "Datasets" and download the dataset into that folder.

``` bash
Datasets\
|------- Zenodo_eeg\
|        |--------- annotations_2017.mat
|        |--------- eeg1.edf
|        |           :
|        |--------- eeg79.edf
|------- processed_data\
         |--------- traindata.npy
         |--------- trainlabels.npy
         |--------- testdata.npy
         |--------- testlabels.npy
```


