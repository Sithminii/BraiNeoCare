# A Patient-Independent Neonatal Seizure Prediction Model Using Reduced Montage EEG and ECG

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

### 01) Create a Conda Environment

If you haven't installed Conda on your PC, please install it by referring to the following link.
[https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Once you have successfully installed Conda on your PC, you can start by creating a Conda environment for the project.
```bash
conda activate my_env
```

### 02) Clone the GitHub repository

Next, clone the GitHub repository by running the following command in your terminal after navigating to a preferred location. Or else you can download the zip from the GitHub repo.

```bash
git clone https://github.com/Sithminii/BraiNeoCare.git
```
Then get into the cloned folder.
```bash
cd BraiNeoCare
```

### 03) Install required libraries
Inside the BraiNeoCare folder, run the following command to install all the required libraries.
```bash
pip install -r requirements.txt
```
### 04) Dataset
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

## Model Training

## Model Interpretability

- For model interpretability, we utilize the SHAP algorithm and obtain scalp maps with seizure localization, as shown below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/28d66f9e-11ae-404e-90d0-25404f95f932" width="474" height="668" />
</p>

### Key Observations

- **Alignment with actual seizure locations:**  
  For most neonates (e.g., Neonates 13, 15, 51, 62, 66, and 77), the model’s predicted high-importance EEG channels closely match the actual seizure locations reported in the Helsinki dataset. Darker connections in the scalp maps indicate higher-importance channels, providing a clear visualization of the model’s decision-making and aiding clinicians in localizing seizure focus.

- **Partial deviations in some cases:**  
  A few neonates (Neonates 11, 21, and 40) show slight deviations between predicted and actual seizure locations. For example, Neonate 11 shows predominant activity in the left hemisphere while the clinical seizure focus is in the right posterior quadrant. These deviations may result from early preictal activity or propagation to neighboring channels.

- **Clinical interpretability and insights:**  
  Despite minor deviations, the scalp plots provide valuable interpretability, showing approximate seizure localization and offering clinicians complementary insights. The visualizations enhance the transparency of the predictive model and can guide further clinical assessment.

## Study Summary

- **Research gap addressed:**  
  Developed a patient-independent seizure prediction model using EEG and ECG signals to improve prediction performance in neonates.

- **Model architecture:**  
  Proposed a lightweight CNN-based model with attention mechanisms, trained and validated on the Helsinki neonatal dataset. The model generalizes well to new subjects with fine-tuning using only a few samples.

- **Explainable AI integration:**  
  Incorporated explainable AI techniques to generate scalp plots from model predictions, helping clinicians visualize and localize upcoming seizures for timely interventions.

- **Clinical impact:**  
  Demonstrates potential for real-world clinical applications, enabling improved neonatal care. Further research with larger datasets is needed to enhance generalization and applicability.

## Acknowledgement

- The computational resources used in this project were funded by the Clair Accelerating Higher Education Expansion and Development (AHEAD) Operation of the Ministry of Higher Education of Sri Lanka, funded by the World Bank.




