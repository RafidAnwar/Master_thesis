import os
import cv2
import scipy.io as sio
import numpy as np

EEG_SRATE=300
GSR_SRATE=4
PPG_SRATE=100

def videolen(vf, fps=30):
    camera = cv2.VideoCapture(vf)
    totalfrms = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    return totalfrms/fps, totalfrms # Return video duration

def cutvideo(vf, target_file, bg_pos, frm_cnt):
    camera = cv2.VideoCapture(vf)
    camera.set(cv2.CAP_PROP_POS_FRAMES, bg_pos) # Set start position for cutting video

    video_format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # Define output codec
    hfile = cv2.VideoWriter()
    filepath = target_file
    frame_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    hfile.open(filepath, video_format, 30, frame_size)

    curtotal = 0
    while camera.isOpened():
        sucess, video_frame = camera.read()
        if sucess is True and curtotal < frm_cnt:
            hfile.write(video_frame)
            curtotal = curtotal + 1
        else:
            hfile.release()
            camera.release()
    print(target_file, 'processing')
    
    

def process4subj(subj, rootpath, savepath):
    data_path = rootpath + '/Raw_data/' + str(subj) + '/sub%d'%(subj) + '/'
    data = sio.loadmat(data_path + 'sig_data.mat')

    eeg = data['eeg_datas']
    eeg_datas = eeg[:, eeg[-1, :] == 40] # Initialize with baseline ID value 40, which is the selected id for rest state

    gsr = data['gsr_datas']
    gsr_datas = gsr[:, gsr[-1, :] == 40]

    ppg = data['ppg_datas']
    ppg_datas = ppg[:, ppg[-1, :] == 40]

    for vid in range(32):
        eeg_cur = eeg[:, eeg[-1, :] == vid]
        gsr_cur = gsr[:, gsr[-1, :] == vid]
        ppg_cur = ppg[:, ppg[-1, :] == vid]
        
        eeg_len = eeg_cur.shape[1]/EEG_SRATE

        gsr_len = gsr_cur.shape[1]/GSR_SRATE
        ppg_len = ppg_cur.shape[1]/PPG_SRATE
        
        vlen, totalfrms = videolen(data_path+'%d.mp4'%(vid))

        data_length = int(min(eeg_len, gsr_len, ppg_len, vlen))

        # Trim data arrays to minimum length
        eeg_cur = eeg_cur[:, -data_length*EEG_SRATE:]
        gsr_cur = gsr_cur[:, -data_length*GSR_SRATE:]
        ppg_cur = ppg_cur[:, -data_length*PPG_SRATE:]

        # Concatenate current trial data to cumulative arrays
        eeg_datas = np.hstack((eeg_datas, eeg_cur))
        gsr_datas = np.hstack((gsr_datas, gsr_cur))
        ppg_datas = np.hstack((ppg_datas, ppg_cur))

        if not os.path.exists(savepath + '%d/'%(subj)):
            os.mkdir(savepath + '%d/'%(subj))

        # Cut and save trial video segment aligned with physiological data
        cutvideo(data_path + '%d.mp4'%(vid), savepath + '%d/'%(subj) + '%d.mp4'%(vid), totalfrms-data_length*30, data_length*30)

    sio.savemat(savepath+'%d/aligned_data.mat'%(subj), {'eeg_datas':eeg_datas, 'gsr_datas':gsr_datas, 'ppg_datas':ppg_datas})
        

# the main function for data alignment
def mainproc(rootpath='./', savepath='./'):
    # process for each subject
    for subj in [1,2,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,28,29,30,32,33,34,35,
                36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,
                63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80]:
        process4subj(subj, rootpath, savepath)
        print(f"participant {subj} done")

if __name__ == '__main__':
    mainproc(savepath='E:/thesis/Aligned_data/')