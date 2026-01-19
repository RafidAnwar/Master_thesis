import os
import numpy as np
from scipy import io as sio
from scipy.signal.windows import hann

from typing import List, Tuple

# Function to calculate PSD features
def extract_psd_feature(raw_data: np.array, sample_freq: int, window_size: int,
                        freq_bands: List[Tuple[int, int]], stft_n=256):
    n_channels, n_samples = raw_data.shape
    
    point_per_window = int(sample_freq * window_size) # Number of samples per window
    window_num = int(n_samples - point_per_window)// int(sample_freq)+1 # Total number of windows
    psd_feature = np.zeros((window_num, len(freq_bands), n_channels))
    
    for window_index in range(window_num):
        start_index, end_index = sample_freq * window_index, sample_freq * window_index + point_per_window
        window_data = raw_data[:, start_index:end_index]
        hdata = window_data * hann(point_per_window)
        fft_data = np.fft.fft(hdata, n=stft_n) # Compute FFT of windowed data
        energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])
        
        for band_index, band in enumerate(freq_bands):
            band_ave_psd = _get_average_psd(energy_graph, band, sample_freq, stft_n)
            psd_feature[window_index, band_index, :] = band_ave_psd
    return psd_feature

# Calculate Differential Entropy (DE) features derived from PSD features for each window and band
def extract_de_feature(raw_data: np.array, sample_freq: int, window_size: int,
                       freq_bands: List[Tuple[int, int]], stft_n=256):
    psd_feature = extract_psd_feature(raw_data, sample_freq, window_size, freq_bands, stft_n)
    return 0.5 * np.log(psd_feature) + 0.5 * np.log(2 * np.pi * np.exp(1) / round(sample_freq / stft_n))

# Calculate average power spectral density (PSD) within a frequency band.
def _get_average_psd(energy_graph, freq_bands, sample_freq, stft_n=256): # Function to calculate average psd
    start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
    end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
    ave_psd = np.mean(energy_graph[:, start_index:end_index+1] ** 2, axis=1)
    return ave_psd
    

def fea4subjs(rootpath):
    for subj in [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33, 34,
                 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]:
        file = rootpath + str(subj) + '/aligned_data.mat'
        data = sio.loadmat(file)
        eeg_datas = data['eeg_datas']

        freq_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 49)] # Define the 5 frequency bands
        data = {}
        vids = []
        de_feas = None
        for i in range(32):
            data[i] = eeg_datas[:, eeg_datas[-1, :] == i]
            de_fea = extract_de_feature(data[i][:-1, :], 300, 5, freq_bands)
            if de_feas is None:
                de_feas = de_fea
            else:
                de_feas = np.concatenate((de_feas, de_fea))
            vids = np.append(vids, np.ones(de_fea.shape[0], np.int32) * i)

        savepath = os.path.join(rootpath, str(subj), 'eegfea_last.mat')
        sio.savemat(savepath, {'feas': de_feas, 'vids': vids})
        print(f"participant {subj} done")

if __name__ == '__main__':
    fea4subjs(rootpath='./Aligned_data/')
