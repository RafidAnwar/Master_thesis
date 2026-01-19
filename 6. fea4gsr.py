import os
from scipy import signal
import numpy as np
from scipy import io as sio
import neurokit2 as nk
from scipy.signal import find_peaks


def butter_lowpass(data, cutfreq, fs, order=3):  # Function for low pass filtering
    cutfreq = cutfreq/(0.5*fs)
    b, a = signal.butter(order, cutfreq, btype='low')
    data_filtered = signal.filtfilt(b, a, data)
    return data_filtered


def statisfeatures(data):  # Function for time domain feature extraction
    fea = {'median': np.median(data), 'mean': np.mean(data), 'std': np.std(data), 'min': np.min(data),
           'max': np.max(data)}
    fea['minratio'] = np.sum(data == fea['min']) / len(data)
    fea['maxratio'] = np.sum(data == fea['max']) / len(data)
    return fea


def freqfeatures(data):  # Function for frequency domain feature extraction
    fre_data = np.fft.fft(data)   # Fast fourier transform for domain conversion
    fre_fea = {'median': abs(np.median(fre_data)), 'mean': abs(np.mean(fre_data)), 'std': abs(np.std(fre_data)),
               'min': abs(np.min(fre_data)), 'max': abs(np.max(fre_data))}
    fre_fea['range'] = abs(fre_fea['max'] - fre_fea['min'])
    return fre_fea


# SCR-specific feature extraction
def extract_scr_features(window_data, fs=4):
    # Find peaks (SCR events)
    peaks, properties = find_peaks(
        window_data,
        height= 0.1, # Minimum amplitude threshold for SCR
        distance=int(fs * 0.5),  # Min .5s between SCRs
        prominence=np.std(window_data) * 0.3
    )

    # Calculate features
    scr_count = len(peaks)
    scr_amps = properties['peak_heights'] if scr_count > 0 else np.array([0])
    scr_rate = scr_count / 5
    auc = np.trapz(window_data)

    return {
        'scr_count': scr_count,
        'scr_rate': scr_rate,
        'scr_mean_amp': np.mean(scr_amps) if scr_count > 0 else 0,
        'scr_max_amp': np.max(scr_amps) if scr_count > 0 else 0,
        'scr_auc': auc
    }


def featurefromgsr(gsr_data, fs=4, wndsize=5, slide=1):
    n_samples = gsr_data.shape[0]
    point_per_window = int(fs*wndsize)
    window_num = int((n_samples - point_per_window) // int(fs * slide))+1
    gsr_data = butter_lowpass(gsr_data, 1.0, 4)  # apply low pass butterworth filter
    feas = None
    for window_index in range(window_num):  # Dividing the data in windows for capturing short term patterns
        start_index, end_index = (fs * slide) * window_index, (fs * slide) * window_index + point_per_window
        window_data = gsr_data[start_index:end_index]  # Extract the GSR data segment for the current window.

        # decompose the windowed gsr_data into tonic & phasic components
        decomposed = nk.eda_phasic(nk.standardize(window_data), sampling_rate=fs, method="cvxeda")
        phasic = decomposed["EDA_Phasic"]
        phasic_feas = statisfeatures(phasic)  # Time domain features of the phasic components

        fea = statisfeatures(window_data)  # compute raw signal time domain features
        fea_1d = statisfeatures(np.diff(window_data))  # 1st order derivative features
        fea_2d = statisfeatures(np.diff(np.diff(window_data)))  # 2nd order derivative features
        fea_freq = freqfeatures(window_data)  # Frequency domain features
        freqs, psd = signal.welch(window_data, 4)  # Power Spectral Density (PSD) using Welch’s method
        avg_psd = np.mean(psd)

        scr_feas = extract_scr_features(phasic, fs)  # Extract SCR specific features from the Phasic components

        fea_row = [fea['median'], fea['mean'], fea['std'], fea['min'], fea['max'], fea['minratio'], fea['maxratio'],
                   fea_1d['median'], fea_1d['mean'], fea_1d['std'], fea_1d['min'], fea_1d['max'], fea_1d['minratio'], fea_1d['maxratio'],
                   fea_2d['median'], fea_2d['mean'], fea_2d['std'], fea_2d['min'], fea_2d['max'], fea_2d['minratio'], fea_2d['maxratio'],
                   fea_freq['median'], fea_freq['mean'], fea_freq['std'], fea_freq['min'], fea_freq['max'], fea_freq['range'],
                   phasic_feas['median'], phasic_feas['mean'], phasic_feas['std'], phasic_feas['min'], phasic_feas['max'], phasic_feas['minratio'], phasic_feas['maxratio'],
                   scr_feas['scr_count'], scr_feas['scr_rate'], scr_feas['scr_mean_amp'], scr_feas['scr_max_amp'], scr_feas['scr_auc'],
                   avg_psd]

        if feas is None:
            feas = np.array(fea_row)
        else:
            feas = np.vstack((feas, fea_row))
    return feas
        

def perifeaext(rootpath):
    invalidSub = []
    for i in [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33, 34,
              35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
              61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]:
        filepath = rootpath + str(i) + '/aligned_data.mat'
        data_org = sio.loadmat(filepath)
        gsr_datas = data_org['gsr_datas']

        feas_gsr = None
        vids = []
        for j in range(32):
            data_gsr = gsr_datas[0, gsr_datas[1, :] == j]
            gsr_fea = featurefromgsr(data_gsr)
            vids = np.append(vids, np.ones(gsr_fea.shape[0], np.int32) * j)
            if feas_gsr is None:
                feas_gsr = gsr_fea
            else:
                feas_gsr = np.vstack((feas_gsr, gsr_fea))

        print(f"participant {i} done")
        savepath = os.path.join(rootpath, str(i), 'gsrfea.mat')
        sio.savemat(savepath, {'feas_gsr': feas_gsr, 'vids': vids})
    print('Invalid subjects:', invalidSub)


if __name__ == '__main__':
    perifeaext(rootpath='./Aligned_data/')
