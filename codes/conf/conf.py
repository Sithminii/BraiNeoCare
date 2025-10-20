def get_config():

    conf = {
        'paths': {
            'zenodo_dataset_path': 'Datasets/Zenodo_dataset',
            'primatry_dataset_path': 'Datasets/processed_data',
            'dataset_path': 'Datasets/processed_data/mfcc_dataset',

            # dataset paths for explainable AI test
            'train_path': 'braineocare/XAI/train_data',
            'test_path': 'braineocare/XAI/test_data',
            'output_path': 'braineocare/XAI/results',
            
            'kfold_output_path':'braineocare/results/zenodo_kfold/checkpts',
            'kfold_fold_path': 'braineocare/results/zenodo_kfold/fold_data',
            'lopo_path': 'braineocare/results/zenodo_lopo',

            'weights_path': 'braineocare/outputs/checkpoint_no_cv.pth',
            'explainer_path': 'braineocare/outputs/shap_explainer.pkl',
            'trained_model_path': 'braineocare/outputs/trained_model.pth',
            'electrode_img_path': 'braineocare/images/electrode_placement.png'
        },

        'train_config': {
            'kfold':  10,
            'batch_size': 256,
            'max_epochs': 300,
            'learning_rate': 0.0004,
            'weight_decay': 0.005,
            'dropout': 0.3,
            'positive_weight': 0.52,
            'reduction': 8
        },

        'data_preprocessing': {
            'filter': {'bandpass': [0.1,70], 'notch':50},
            'segmenting': {
                # Durations in seconds
                'segment_duration': 5,
                'preictal_duration': 30*60,
                'postictal_duration': 30*60,
                'int_end_to_ict_duration': 60*60
            },
            'mfcc': {
                'n_mfcc': 20,
                'n_mels': 20,
                'hop_length': 128,
                'f_min': 0.1,
                'f_max': 100
            }
        }

    }

    return conf