#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
import scipy.io as scp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from asrpy import ASR
matplotlib.use('TkAgg')
get_ipython().run_line_magic('matplotlib', 'notebook')

def Preprocessing(l_freq_eog, h_freq_eog, l_freq_eeg, h_freq_eeg):
    
    # 1. File Paths for GDF Files
    file_paths = ['A01T.gdf', 'A02T.gdf', 'A03T.gdf', 'A04T.gdf', 'A05T.gdf', 
                  'A06T.gdf', 'A07T.gdf', 'A08T.gdf', 'A09T.gdf']
    all_data = []
    all_labels = []
    
    # 2. Loop over Files to Process Each One
    for file_path in file_paths:
        
        # Add a headline with the EEG file name
        print("\n" + "="*30)
        print(f"Preprocessing file: {file_path}")
        print("="*30 + "\n")
        
        # Load GDF file and print channel names
        raw = mne.io.read_raw_gdf(file_path, preload=True)
        print(raw.info['ch_names'])
        
        # 3. Channel Renaming and Setting Types
        mapping = {
            'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 
            'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz',
            'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 
            'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-Pz': 'Pz',
            'EEG-15': 'P2', 'EEG-16': 'POz', 'EOG-left': 'EOG1', 'EOG-central': 'EOG2', 
            'EOG-right': 'EOG3'
        }
        raw.rename_channels(mapping)
        
        # Separate EEG and EOG channels
        eog_channels = ['EOG1', 'EOG2', 'EOG3']
        eeg_channels = [ch for ch in raw.ch_names if ch not in eog_channels]
        raw.set_channel_types({ch: 'eog' for ch in eog_channels})
        raw.set_channel_types({ch: 'eeg' for ch in eeg_channels})
        
        # 4. Set Montage (Channel Positioning)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='warn')
        
        # Plot sensor locations
        raw.plot_sensors(show_names=True)
        
        # 5. Filter EEG and EOG Channels
        raw.filter(l_freq_eeg, h_freq_eeg, picks=eeg_channels)
        raw.filter(l_freq_eog, h_freq_eog, picks=eog_channels)
        
        
        # 6. EEG and EOG picks selecction
        eog_picks = mne.pick_types(raw.info, eeg=False, eog=True)
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
        
        # Initialize ICA
        ica_eeg = mne.preprocessing.ICA(n_components=22, random_state=0)
        picks = mne.pick_types(raw.info, eeg=True, eog=True)
        ica_eeg.fit(raw, picks=picks)
        
        # Find and exclude bad components related to EOG artifacts
        eog_indices, scores = ica_eeg.find_bads_eog(raw, ch_name=eog_channels)
        ica_eeg.plot_scores(scores)
        ica_eeg.plot_sources(raw, show_scrollbars=False)
        ica_eeg.exclude = eog_indices
        
        # Apply the ICA solution to the raw data
        raw_clean = ica_eeg.apply(raw.copy(), exclude=ica_eeg.exclude)
        
        # Extract only EEG channels
        raw_clean_eeg = raw_clean.copy().pick_channels(eeg_channels)
        
        # 7. ASR for Noise Removal
        asr = ASR(sfreq=raw_clean_eeg.info['sfreq'])
        clean_segment = raw_clean_eeg.copy().crop(tmin=760, tmax=820)
        asr.fit(clean_segment)
        raw_clean_eeg = asr.transform(raw_clean_eeg)
        
        # Save cleaned data
        raw_clean_eeg.save('cleaned_raw-epo_No_epoching_before_ICA.fif', overwrite=True)
        raw_clean_eeg.plot(title='Raw Data After ICA&ASR')
        
        # 8. Plot Power Spectral Density (PSD) Before and After ICA
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        raw.plot_psd(ax=ax[0], fmax=50, show=False, color='r')
        ax[0].set_title('PSD Before ICA&ASR')
        raw_clean_eeg.plot_psd(ax=ax[1], fmax=50, show=False, color='g')
        ax[1].set_title('PSD After ICA&ASR')
        plt.tight_layout()
        plt.show()
        
        # 9. Epoch Cleaned Data (Events of Interest)
        events_clean, _ = mne.events_from_annotations(raw_clean_eeg)
        selected_event_id = {'left_hand': 7, 'right_hand': 8, 'foot': 9, 'tongue': 10}
        
        # Specific case for file A04T.gdf
        if file_path == 'A04T.gdf':
            selected_event_id = {'left_hand': 5, 'right_hand': 6, 'foot': 7, 'tongue': 8}
        
        filtered_events = np.array([event for event in events_clean if event[2] in selected_event_id.values()])
        epochs = mne.Epochs(raw_clean_eeg, filtered_events, selected_event_id, tmin=-0.5, tmax=4, baseline=(-0.5, 0), preload=True, event_repeated='merge')
        epochs.plot()
        
        # 10. Extract Data and Labels
        X = epochs.get_data()
        print(X.shape)  # (n_epochs, n_channels, n_times)
        Y = epochs.events[:, 2]
        Y = np.array(Y)
        
        # Map labels to 0, 1, 2, 3
        labels_Y = {7: 0, 8: 1, 9: 2, 10: 3}
        if file_path == 'A04T.gdf':
            labels_Y = {5: 0, 6: 1, 7: 2, 8: 3}
        Y = np.vectorize(labels_Y.get)(Y)
        
        # Append data and labels to lists
        all_data.append(X)
        all_labels.append(Y)
    
    # 11. Concatenate All Data and Labels Across Subjects
    X_all_No_Epoching_Before_ICA = np.concatenate(all_data, axis=0)
    Y_all_No_Epoching_Before_ICA = np.concatenate(all_labels, axis=0)
    
    #Save the preprocessed data
    np.save('X_all_No_Epoching_Before_ICA.npy', X_all_No_Epoching_Before_ICA)
    np.save('Y_all_No_Epoching_Before_ICA.npy', Y_all_No_Epoching_Before_ICA)
    

    # Return the concatenated data and labels
    return X_all_No_Epoching_Before_ICA, Y_all_No_Epoching_Before_ICA

