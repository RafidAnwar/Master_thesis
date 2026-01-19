import os
from scipy import signal
import numpy as np
from scipy import io as sio
import neurokit2 as nk


def butter_bandpass(data, lowcut, highcut, fs, order=3):  # Function for band pass filtering
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    data_filtered = signal.filtfilt(b, a, data)
    return data_filtered


def statisfeatures(data):  # Function for time domain feature extraction
    fea = {'median': np.median(data), 'mean': np.mean(data), 'std': np.std(data), 'min': np.min(data),
           'max': np.max(data)}
    fea['minratio'] = np.sum(data == fea['min']) / len(data)
    fea['maxratio'] = np.sum(data == fea['max']) / len(data)
    return fea


def freqfeatures(data):  # Function for frequency domain feature extraction
    fre_data = np.fft.fft(data)  # Fast fourier transform for domain conversion
    fre_fea = {'median': abs(np.median(fre_data)), 'mean': abs(np.mean(fre_data)), 'std': abs(np.std(fre_data)),
               'min': abs(np.min(fre_data)), 'max': abs(np.max(fre_data))}
    fre_fea['range'] = abs(fre_fea['max'] - fre_fea['min'])
    return fre_fea


def hrv_feature(ppg_data, fs=100):  # HRV Feature extraction function

    ppg_clean = nk.ppg_clean(ppg_data, fs)   # preparing raw ppg signal for systolic peak detection
    signals, info = nk.ppg_peaks(ppg_clean, fs, correct_artifacts=True)  # Find systolic peaks in the cleaned ppg signal
    peak = info['PPG_Peaks']
    if len(peak) <= 2:  # Atleast 2 peaks required for calculation of hrv feature indices via neurokit2
        print('no peak detected')
        return {'mean': 0, 'median': 0, 'RMSSD':0 , 'SDNN': 0, 'SDRMSSD': 0, 'pNN20': 0, 'hti': 0}

    hrv_time = nk.hrv_time(peak, fs)  # Computing time-domain indices of HRV
    return {
        # Selected Time Domain features:
        'mean': hrv_time["HRV_MeanNN"].iloc[0],
        'median': hrv_time["HRV_MedianNN"].iloc[0],
        'RMSSD': hrv_time["HRV_RMSSD"].iloc[0],  # square root of the mean of the squared successive differences
        'SDNN': hrv_time["HRV_SDNN"].iloc[0],  # Standard deviation of the peaks
        'SDRMSSD': hrv_time["HRV_SDRMSSD"].iloc[0] if not np.isnan(hrv_time["HRV_SDRMSSD"].iloc[0]) else 0, # SDNN/RMSSD
        'pNN20': hrv_time["HRV_pNN20"].iloc[0],  # percentage of absolute differences in successive peaks > 20 ms
        'hti': hrv_time["HRV_HTI"].iloc[0]  # HRV triangular index
        }


def featurefromppg(ppg_data, fs=100, wndsize=5, slide=1):
    n_samples = ppg_data.shape[0]
    point_per_window = int(fs * wndsize)
    window_num = int((n_samples - point_per_window) // int(fs * slide)) + 1
    ppg_data = butter_bandpass(ppg_data, 0.5, 8, 100) # apply Band pass butterworth filter

    feas = None
    for window_index in range(window_num):  # Dividing the data in windows for capturing short term patterns
        start_index, end_index = (fs * slide) * window_index, (fs * slide) * window_index + point_per_window
        window_data = ppg_data[start_index:end_index]

        fea = statisfeatures(window_data)  # compute time domain features
        fea_1d = statisfeatures(np.diff(window_data))  # 1st order derivative features
        fea_2d = statisfeatures(np.diff(np.diff(window_data)))  # 2nd order derivative features
        fea_freq = freqfeatures(window_data)  # Frequency domain features

        hrv = hrv_feature(window_data, fs)  # Get heart rate variability features from defined function

        fea_row = [fea['median'], fea['mean'], fea['std'], fea['min'], fea['max'], fea['minratio'], fea['maxratio'],
                   fea_1d['median'], fea_1d['mean'], fea_1d['std'], fea_1d['min'], fea_1d['max'], fea_1d['minratio'], fea_1d['maxratio'],
                   fea_2d['median'], fea_2d['mean'], fea_2d['std'], fea_2d['min'], fea_2d['max'], fea_2d['minratio'], fea_2d['maxratio'],
                   fea_freq['median'], fea_freq['mean'], fea_freq['std'], fea_freq['min'], fea_freq['max'], fea_freq['range'],
                   hrv['mean'], hrv['median'], hrv['RMSSD'], hrv['SDNN'], hrv['SDRMSSD'], hrv['pNN20'], hrv['hti']]
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
        ppg_datas = data_org['ppg_datas']

        feas_ppg = None
        vids = []
        for j in range(32):
            data_ppg = ppg_datas[0, ppg_datas[1, :] == j]
            ppg_fea = featurefromppg(data_ppg)
            vids = np.append(vids, np.ones(ppg_fea.shape[0], np.int32) * j)
            if feas_ppg is None:
                feas_ppg = ppg_fea
            else:
                feas_ppg = np.vstack((feas_ppg, ppg_fea))

        print(f"participant {i} done")
        savepath = os.path.join(rootpath, str(i), 'ppgfea.mat')
        sio.savemat(savepath, {'feas_ppg': feas_ppg, 'vids': vids})
    print('Invalid subjects:', invalidSub)


if __name__ == '__main__':
    perifeaext(rootpath='./Aligned_data/')
