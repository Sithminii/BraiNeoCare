import os
import numpy as np
import scipy
import mne
import torch
import torchaudio

from src.dataset.initialize_data import read_file, remove_artefacts, create_data_arrays

mne.set_log_level('ERROR')

# Extracts the numerical part from a string
def get_ID(string: str):
    string_list = list(string)
    idx=''
    for i in string_list:
        if i.isdigit():
            idx+=i
    return int(idx)



def check_consecative_zeros(array:np.array, sfreq:float, thresh:float=1e-6):
    '''
    inputs:
        array - signal as a 1D array
        sfreq - sampling frequency of the signal
        thresh - threshold voltage to detect as a flat line
    output:
        returns True if any 1-second long flat line was detected, otherwise returns False
    '''
    diff = np.diff(array)
    zero_ids = np.where(np.abs(diff) <= thresh)
    zero_ids = zero_ids[0]    

    # Check 1-second segments of the original sample
    seg_size = int(1*sfreq)
    
    for k in zero_ids:
        seg = array[k:k+seg_size]
        
        # Check if the selected segment is a flat line
        if np.all(np.abs(seg) <= thresh):
            return True
        else:
            continue
    
    return False



def find_seizure_time(file_no, annotations):
    
    '''
    inputs:
        file_no - Number of the neonate
        annotations - matlab annotation data
    output:
        s_time - numpy array containing the start time of each seizure event in the recording in seconds
        e_time - numpy array containing the end time of each seizure event in the recording in seconds
    '''
    
    annotations_by_consensus = annotations['annotat_new'][0][file_no-1][0] & annotations['annotat_new'][0][file_no-1][1] & annotations['annotat_new'][0][file_no-1][2]

    a = np.where(annotations_by_consensus==1)[0]
    start_time = []
    end_time = []
    if len(a) != 0:
        start_time.append(a[0])  
        for r in range(1,a.shape[0]):
            if a[r]-a[r-1]!=1:
                end_time.append(a[r-1])
                start_time.append(a[r]) 
        end_time.append(a[-1])

    s_time = np.array(start_time)
    e_time = np.array(end_time)   
    return s_time,e_time



def find_seizure_recordings(data_folder, annotations):
    '''
    inputs:
        data_folder - path of the folder containing signal recordings
        annotations - matlab annotation data
    output:
        returns a list of file IDs for recordings with seizures annotated by all three experts
    '''
    seizure_files = []
    files_list = [os.path.join(data_folder,file) for file in os.listdir(data_folder) if file.endswith('.edf')]
    files = sorted(files_list, key=get_ID)

    for file in files:
        n = get_ID(file)
        s_time,_ = find_seizure_time(n, annotations)
        if len(s_time) != 0:
            seizure_files.append(file)
    return seizure_files



def compute_mfcc(data:np.array, sfreq:int, mfcc_specs:dict):

    '''
    inputs:
        data - 2d array of the signals; (n_features, n_samples)
        sfreq - sampling frequency of the signals
        mfcc_specs - dictionary containing specifictions for computing MFCCs 
    
    output:
        MFCC matrices of the multichannel signal
        output_shape; (n_features, n_mels, time_segments)        
    '''

    n_mels = mfcc_specs['n_mels']
    n_mfcc = mfcc_specs['n_mfcc']
    hop_length = mfcc_specs['hop_length']
    f_min = mfcc_specs['f_min']
    f_max = mfcc_specs['f_max']
    
    tensor_data = torch.tensor(data, dtype=torch.float)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sfreq,
                                                n_mfcc=n_mfcc,
                                                log_mels = False,
                                                melkwargs={"n_fft": sfreq,
                                                           "n_mels": n_mels,
                                                           "f_min":f_min, 
                                                           "f_max":f_max,
                                                           "hop_length":hop_length})
    mfcc = mfcc_transform(tensor_data)
    return mfcc.numpy()



def read_and_extract_data(data_folder:str,
                          dest_path:str,
                          annotation_file:str,
                          filter_data:dict,
                          duration_data:dict,
                          mfcc_specs:dict):

    '''
    inputs:
        data_folder - path of the folder containing signal recordings
        dest_path - path to save the outputs
        annotation_file - path to the matlab annotations file
        filter_data - dictionary containing denoising filter specifictaions
        duration_data - dictionary containing important durations for extracting and segmenting interictal and preictal data

    Saves neonate-wise interictal and preictal samples in .txt format and MFCC matrices in .npy format separately
    '''
    
    os.makedirs(dest_path, exist_ok=True)
    
    # Create two folders to store extracted signal samples and computed MFCC matrices separately
    signal_sample_dest = os.path.join(dest_path,'raw_data')
    mfcc_dest = os.path.join(dest_path, 'mfcc_dataset')
    
    os.makedirs(signal_sample_dest, exist_ok=True)
    os.makedirs(mfcc_dest, exist_ok=True)

    # Read matlab annotations file
    annotations = scipy.io.loadmat(annotation_file)
    
    # Define channels to retain
    ch_names = ['C3', 'C4', 'Cz', 'Fp1', 'Fp2', 'O1', 'O2', 'T3', 'T4', 'ECG']

    files = find_seizure_recordings(data_folder, annotations)
    
    # Extract the filter specifications
    bandpass_filter = filter_data['bandpass']
    notch_filter = filter_data['notch']

    # Extract the important durations for segmenting EEG and ECG arrays
    bound_tolerence = duration_data['boundary']
    seg_duration = duration_data['segment_size']
    preic_durat = duration_data['preictal_duration']
    post_durat = duration_data['postictal_duration']
    interictal_ictal_gap = duration_data['int_end_to_ict_duration']


    counter_i = 0
    counter_p = 0
    
    # Iterate over each recording
    for file in files:
        # Get seizure start and end time
        n = get_ID(file)
        s_time,e_time = find_seizure_time(n, annotations)      

        print(f'Processing Neonate {n}...')
        
        preic_perfile = 0
        interic_perfile = 0
        
        # Read the edf file and generate required channels
        rawdata = read_file(file, ch_names)
        x_clean, sfreq = remove_artefacts(rawdata, bandpass=bandpass_filter, notch=notch_filter)
        signal = create_data_arrays(x_clean, ch_names, ecg_first=False)

        seizure = 0

        for s in range(len(s_time)):
            seizure += 1

            preictal_signals = []
            interictal_signals = []
            end_interictal_signals = []

            #preictal extraction        
            u_bound = s_time[s] * sfreq        
            l_bound = (s_time[s] - preic_durat) * sfreq

            l_bound += (u_bound - l_bound) % (seg_duration*sfreq)

            # Define lower bound limit
            if s == 0:                                              # beginning of the signal
                l_bound_min = bound_tolerence * sfreq
            else:                                                   # intermediate seizures
                l_bound_min = (e_time[s-1] + post_durat)*sfreq
                

            #Extracting segments
            if l_bound_min <= l_bound:
                
                # extract preictal data
                preictal_segment = signal[:, l_bound:u_bound]
                n_samples = int((preictal_segment.shape[1])/(seg_duration*sfreq))
                partitions = np.split(preictal_segment, n_samples, axis=1)

                preictal_signals.extend(partitions)
                
                # interictal data extraction
                u_bound = (s_time[s] - interictal_ictal_gap) * sfreq
                l_bound_min += (u_bound - l_bound_min) % (seg_duration*sfreq)
                
                l_bound = l_bound_min
    
                if l_bound < u_bound:
                    interictal_segment = signal[:, l_bound:u_bound]
                    n_samples = int((interictal_segment.shape[1])/(seg_duration*sfreq))
                    partitions = np.split(interictal_segment, n_samples, axis=1)
    
                    interictal_signals.extend(partitions)
                else:
                    pass

            # Cases when the full preictal duration is not available
            elif l_bound_min < u_bound:
                l_bound_min += (u_bound - l_bound_min) % (seg_duration*sfreq)
                
                if l_bound_min < u_bound:
                    # Extract possible perictal data
                    print('extracting preictal data')
                    preictal_segment = signal[:, l_bound_min:u_bound]
                    n_samples = int((preictal_segment.shape[1])/(seg_duration*sfreq))
                    partitions = np.split(preictal_segment, n_samples, axis=1)

                    preictal_signals.extend(partitions)
                else:pass
            else:
                pass

            # Extract interictal data from end of the signal
            if s == len(s_time)-1:
                l_bound = (e_time[s] + post_durat)*sfreq
                u_bound = ((signal.shape[1]//sfreq) - bound_tolerence) * sfreq

                l_bound += (u_bound - l_bound) % (seg_duration*sfreq)
                
                if l_bound < u_bound:
                    print('extracting interictal data')
                    interictal_segment = signal[:, l_bound:u_bound]
                    n_samples = int((interictal_segment.shape[1])/(seg_duration*sfreq))
                    partitions = np.split(interictal_segment, n_samples, axis=1)
    
                    end_interictal_signals.extend(partitions)
                else:
                    pass
            else:
                pass


            # Check if there are data samples to store for the current neonate
            if len(preictal_signals)>=10 and len(interictal_signals)>=10:
                
                # Create a folder for the current neonate to store signal samples
                sample_folder_name = os.path.join(signal_sample_dest,f'neon_{n}')
                os.makedirs(sample_folder_name, exist_ok=True)

                # Create a folder for the current neonate to store MFCC matrices
                mfcc_folder_name = os.path.join(mfcc_dest,f'neon_{n}')
                os.makedirs(mfcc_folder_name, exist_ok=True)

                # Save preictal data samples
                for sample in preictal_signals:
                    if not check_consecative_zeros(sample[0], sfreq, thresh=1e-6):
                        # Save the signal sample
                        np.savetxt(os.path.join(sample_folder_name,f'{counter_p}_pre.txt'), sample)

                        # Compute and save the MFCC matrices of the sample
                        mfcc = compute_mfcc(sample, sfreq, mfcc_specs)
                        np.save(os.path.join(mfcc_folder_name,f'{counter_p}_pre.npy'), mfcc)
                        
                        preic_perfile += 1
                        counter_p += 1
                    else:
                        print('A flat line detected and dropped!')
                        continue

                # Save interictal data samples   
                for sample in interictal_signals:
                    if not check_consecative_zeros(sample[0], sfreq, thresh=1e-6):
                        np.savetxt(os.path.join(sample_folder_name,f'{counter_i}_int.txt'), sample)

                        # Compute and save the MFCC matrices of the sample
                        mfcc = compute_mfcc(sample, sfreq, mfcc_specs)
                        np.save(os.path.join(mfcc_folder_name,f'{counter_i}_int.npy'), mfcc)

                        interic_perfile += 1
                        counter_i += 1
                    else:
                        print('A flat line detected and dropped!')
                        continue
                
                # Save interictals from end of the signal
                if len(end_interictal_signals)!=0:

                    # Save data
                    for sample in end_interictal_signals:
                        if not check_consecative_zeros(sample[0], sfreq, thresh=1e-6):
                            np.savetxt(os.path.join(sample_folder_name,f'{counter_i}_int.txt'), sample)

                            # Compute and save the MFCC matrices of the sample
                            mfcc = compute_mfcc(sample, sfreq, mfcc_specs)
                            np.save(os.path.join(mfcc_folder_name,f'{counter_i}_int.npy'), mfcc)

                            interic_perfile += 1
                            counter_i += 1
                        else:
                            print('A flat line detected and dropped!')
                            continue
            else:
                pass

        print(f'interictals = {interic_perfile}')
        print(f'preictals = {preic_perfile}')
        print('-'*100,'\n')