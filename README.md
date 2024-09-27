# EEG Signal Processing Pipeline for Enhanced Model Performance


## Overview

This project contains implementable Python and Jupyter Notebook codes to describe the pipeline of EEG signal processing, focusing on different preprocessing techniques to illustrate the importance of preprocessing in achieving good model performance. A key feature of this project is the introduction of CNN-LSTM-CNN-LSTM model architecture, which demonstrates innovative approaches to EEG signal classification. In Experiment 5, we observe the significant impact of bias initialization on the model performance in deep learning.

Below is a description of the contents of each folder:

1. Experiment 1: Preprocessing Without ASR

Purpose: This experiment processes EEG data without Artifact Subspace Reconstruction (ASR), using only Independent Component Analysis (ICA).

2. Experiment 2: Preprocessing with ICA and ASR

Purpose: This experiment preprocesses the EEG data using ICA on all training files, with ASR applied only to the most noisy files (A04T & A09T).

3. Experiment 3: No Epoching Before ICA Preprocessing

Purpose: In this experiment, the EEG data is processed without epoching before applying ICA to compare the effects of epoching on model performance.

4. Experiment 4: ICA and ASR Preprocessing

Purpose: EEG data is processed using both ICA and ASR techniques for all training files, achieving better performance on the fixed CNN-LSTM-CNN-LSTM model compared to the previous techniques.

5. Experiment 5: ICA, ASR, and LSTM Forget Gate Bias Initialization

Purpose: This experiment extends the preprocessing with ICA and ASR on all training files, investigating the effect of forget gate bias initialization on the model. Here, we explore the difference in performance when the forget gate bias initialization is set to 0 instead of the default 1.


## BCI Dataset

In this project the used dataset was "BCI Competition 2008– Graz data set A".

- **Dataset Description Link**:https://www.bbci.de/competition/iv/desc_2a.pdf
- **Download Link**:https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit


## Requirements

To run this project, the following software versions are required:

- **Python**: 3.8.3
- **mne**: 1.6.1
- **PyTorch**: 2.0.1


## Final Results

![alt text] (https://github.com/MohamedRasmyy/EEG-Signal-Processing-Pipeline-for-Enhanced-Model-Performance/blob/EEG-Signal-Processing-Pipeline-for-Enhanced-Model-Performance/Comparison%20of%20Train,%20Validation,%20and%20Test%20Accuracy%20on%20different%20initialization%20values%20for%20%20bias.PNG?raw=true)

! [alt text] (Comparison of Train, Validation, and Test Accuracy on different initialization values for  bias.png)


## Miscellaneous

Please send any questions you might have about the code  to mohamedrasmi689@gmail.com
