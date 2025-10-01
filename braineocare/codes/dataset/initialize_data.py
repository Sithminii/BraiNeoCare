import numpy as np
import mne

mne.set_log_level('ERROR')


def read_file(file_path:str, ch_names:list):   
    '''
    inputs:
        file_path - path to the .edf file
        ch_names - channels to retain
    output:
        returns the signal as an mne object
    '''  
    # Read .edf file
    raw_data = mne.io.read_raw_edf(file_path, preload=True)
    
    # Rename channels for consistency
    rename_dict = {}
    for ch in raw_data.ch_names:
        if ch.startswith('EEG '):
            standard_name = ch.replace('EEG ', '').replace(ch[-4:], '')
            rename_dict[ch] = standard_name
        elif 'ECG' in ch:
            rename_dict[ch] = 'ECG'
        elif 'Resp' in ch:
            rename_dict[ch] = 'Resp'
    
    raw_data.rename_channels(rename_dict)
    raw_data.set_channel_types({'ECG': 'ecg', 'Resp': 'misc'})
    
    raw_data.set_montage('standard_1020')

    # Extract required channels
    raw_data = raw_data.pick_channels(ch_names, ordered=True)

    return raw_data



def remove_artefacts(x_mne, bandpass:list, notch:float):
    '''
    inputs:
        x_mne - mne object of the noisy signal
        bandpass - cutoff frequencies of the 4th order butterworth bandpass filter; [l_freq,h_freq]
        notch - bandstop frequency of the 4th order butterworth notch filter
    output:
        x_clean - artefact removed signal as an mne object
        sfreq - sampling frequency of the signal
    '''

    assert len(bandpass) == 2 and bandpass[0] < bandpass[1]

    x_clean = x_mne.copy().notch_filter(freqs=notch, method='iir', verbose=True)
    x_clean = x_clean.copy().filter(l_freq=bandpass[0], h_freq=bandpass[1], method='iir', verbose=True)

    sfreq = int(x_clean.info['sfreq'])

    return x_clean, sfreq
   


def get_channel_index(x_mne, ch_names:list):
    '''
    inputs:
        x_mne - signal as an mne object
        ch_names - retained channels
    output:
        returns a dictionary containing the indexwise location of each EEG channel in the signal array
    '''

    index_dict = {}
    data_chs = x_mne.info['ch_names']
    for ch_tag in ch_names:
        # Add location to the dictionary
        index_dict[ch_tag] = data_chs.index(ch_tag) 
    
    return index_dict



def normalize_data(x:np.array, mode='zeromean', eps:float=1e-8):
    '''
    inputs:
        x - EEG and ECG signals as a 2d numpy array; (n_channels, n_samples)
        mode - 'minmax' or 'zeromean'
    output:
        returns channel-wise normalized 2d signal array
    '''
    if mode == 'zeromean':
        # Channel-wise mean and standard deviation
        mean = np.mean(x, axis=1).reshape(-1,1)
        std = np.std(x, axis=1).reshape(-1,1)

        # Zero-mean normalization
        x_norm = (x - mean)/(std + eps)

    elif mode == 'minmax':
        # Get chanel-wise minimum and maximum values
        minimum =  np.min(x, axis=1).reshape(-1,1)
        maximum = np.max(x, axis=1).reshape(-1,1)

        # Min-max normalization
        x_norm = (x - minimum)/(maximum - minimum + eps)
    else:
        raise ValueError(f'Mode should be either "zeromean or "minmax", got {mode}')
    return x_norm



def create_data_arrays(x_mne, ch_names:list, ecg_first=False): 
    '''
    inputs:
        x_mne - signal as an mne object
        ch_names - retained channels in the required order; [ecg_channel, *eeg_channels] or [*eeg_channels, ecg_channel]
        ecg_first - True if the ecg channel is in the beginning otherwise False; Default Fault
    output:
        returns signals as a 2d numpy array; (n_features, n_samples) where n_features=19
    '''

    if ecg_first:
        ch_ecg = [ch_names[0]]
        ch_indices = get_channel_index(x_mne, ch_names[1:])
    else:
        ch_ecg = [ch_names[-1]]
        ch_indices = get_channel_index(x_mne, ch_names[:-1])

    ecg_signal = x_mne.copy().pick_channels(ch_ecg).get_data()
    data_ = x_mne.get_data()

    # Separate the channels
    C3  = data_[ch_indices['C3']]
    C4  = data_[ch_indices['C4']]
    CZ  = data_[ch_indices['Cz']]
    Fp1 = data_[ch_indices['Fp1']]
    Fp2 = data_[ch_indices['Fp2']]
    O1  = data_[ch_indices['O1']]
    O2  = data_[ch_indices['O2']]
    T3  = data_[ch_indices['T3']]
    T4  = data_[ch_indices['T4']]

    # Generate relative signals
    eegecg_array = np.zeros((19, data_.shape[1]))

    relative_signals = [Fp1-T3, T3-O1, Fp1-C3, C3-O1, Fp2-C4, C4-O2,
                        Fp2-T4, T4-O2, T3-C3, C3-CZ, CZ-C4, C4-T4,
                        Fp1-CZ, CZ-O1, Fp2-CZ, CZ-O2, Fp1-O2, Fp2-O1]
    
    eegecg_array[:-1] = np.array(relative_signals)
    eegecg_array[-1] = ecg_signal

    # Normalize data
    eegecg_array = normalize_data(eegecg_array, mode='minmax')
    
    return eegecg_array
