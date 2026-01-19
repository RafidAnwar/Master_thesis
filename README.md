# Leveraging Multimodal Dataset for Enhancing Mixed Emotion Recognition

---
## Usage Instructions

- **MATLAB Requirements**  
  - MATLAB R2024b  
  - EEGLab v2025.0.0 (for EEG processing)

- **Python Requirements**  
  - Python 3.10  
  - Essential libraries: see `requirements.txt` for all the essential dependencies 
  - The pre trained model used for transfer learning in emo_affectnet.py was develped by ryumina et al. and is available at "https://github.com/ElenaRyumina/EMO-AffectNetModel"

- **Execution Order**  
  - Run the scripts in "code_Leveraging_Multimodal_Dataset_for_Enhancing_Mixed_Emotion_Recognition" folder sequentially as outlined below to achieve optimal pipeline results.
---
## Pipeline Structure

### 1. PreprocessEEG.m: 
Preprocesses raw EEG signas by applies re-referencing, filtering, wavelet denoising and ICA for artifact removal using Matlab.  
### 2. Phy_trial_seperate.m:
Segments the EEG, GSR, and PPG signals into discrete trial-wise samples.  
### 3. Format_video.py: 
Splits each subject's video into trial-matched subclips for subsequent alignment.
### 4. AlignAllData.py: 
Aligns all signals (EEG, GSR, PPG, facial video) using trial end-points as synchronization anchors.
### 5. fea4eeg.py: 
Extracts *differential entropy* (DE) features from five EEG frequency bands: delta (1–3 Hz), theta (4–7 Hz), alpha (8–13 Hz), beta (14–30 Hz), gamma (31–50 Hz); across 18 channels.  
### 6. fee4gsr.py: 
After preprocessing, computes comprehensive time and frequency domain features from both preprocessed GSR signals and its Phasic components.  
### 7. fee4ppg.py: 
After preprocessing, extracts PPG features (time/frequency domain) from the preprocessed ppg signals and heart rate variability (HRV) features from its detected systolic peaks.  
### 8. emo_affectnet.py: 
Utilizes MediaPipe for 3D face mesh extraction, feeding video frames to a pretrained Emo-AffectNet model for facial feature extractions.
### 9. feature_fusion.py: 
Concatenates EEG, GSR, PPG, and facial features to generate 676 dimension feature vector.
### 10. lstm_modality.py: 
Implements ‘leave-one-subject-out’ cross-validation with two layered LSTM model architecture for the three class emotion classification.
### 11. xai.py: 
Performs modality perturbation and uncertainty quantification using the saved models in the previous step.



