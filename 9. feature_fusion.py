import scipy.io as sio
import numpy as np

feas, vids, label, feas_list, vids_list, label_list,parti_list  = [], [], [], [], [], [], []

for sub in [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33, 34,
                 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]:
    path = './Aligned_data/' + str(sub) + '/'
    eeg_fea = sio.loadmat(path + 'eegfea.mat')  # Load the EEG data
    gsr_fea = sio.loadmat(path + 'gsrfea.mat')  # Load the GSR data
    ppg_fea = sio.loadmat(path + 'ppgfea.mat')  # Load the PPG data
    video_fea = sio.loadmat(path + 'videoFea_affectnet.mat')  # Load the Video data
    for vid in range(32):
        # Cut the video for samples from every participant
        eeg_f = eeg_fea['feas'][(eeg_fea['vids'].flatten() == vid)]
        r,c,p = eeg_f.shape
        eeg_f = eeg_f.reshape(r, c*p)
        gsr_f = gsr_fea['feas_gsr'][(gsr_fea['vids'].flatten() == vid)]
        ppg_f = ppg_fea['feas_ppg'][(ppg_fea['vids'].flatten() == vid)]
        video_f = video_fea['feas_all'][(video_fea['vids'].flatten() == vid)]
        length = int(video_f.shape[0] - gsr_f.shape[0])
        feature = np.hstack([eeg_f[4:,:], gsr_f, ppg_f, video_f[length:,:]])
        video = np.ones((feature.shape[0], 1))*vid
        parti = np.ones((feature.shape[0], 1))*sub

        if vid <= 7:
            lab = np.ones((feature.shape[0], 1)) * 0
        elif 7 < vid <= 15:
            lab = np.ones((feature.shape[0], 1)) * 1
        else:
            lab = np.ones((feature.shape[0], 1)) * 2

        feas_list.append(feature)
        vids_list.append(video)
        label_list.append(lab)
        parti_list.append(parti)

    print(f'participant {sub} done')

feas = np.vstack(feas_list)
vids = np.vstack(vids_list)
label = np.vstack(label_list)
participant = np.vstack(parti_list)
sio.savemat('./' + 'full_features_affectnet_last.mat', {'feas': feas, 'vids': vids, 'label': label, 'participant': participant})
