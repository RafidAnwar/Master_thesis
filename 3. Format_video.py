import cv2
import os
import csv

def videocilp(vf, path4save, vid, bg_time, ed_time, fps=30):
    camera = cv2.VideoCapture(vf)
    bg_frame = int(bg_time*fps)
    total_frames = int((ed_time-bg_time)*fps)
    
    camera.set(cv2.CAP_PROP_POS_FRAMES, bg_frame)

    video_format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # Define output video codec
    hfile = cv2.VideoWriter()
    filepath = path4save + str(vid) + '.mp4'
    frame_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    hfile.open(filepath, video_format, 30, frame_size)

    curtotal = 0
    while camera.isOpened():
        sucess, video_frame = camera.read()
        if sucess is True and curtotal < total_frames:
            hfile.write(video_frame)
            curtotal = curtotal +1
        else:
            hfile.release()
            camera.release()


def mainproc(rootpath):
    for subj in [1,2,5,6,7,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,28,29,30,32,33,34,35,
                36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,
                63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80]:
        path4save = rootpath + str(subj) + '/sub' + str(subj) + '/'
        v_path = rootpath + str(subj) + '/' + str(subj) + '/'
        vnames = [v for v in os.listdir(v_path) if v.endswith('_new.mp4')]
        if len(vnames) < 1:
            vnames = [v for v in os.listdir(v_path) if v.endswith('.mp4')]
        vname= vnames[0]
        
        # Read timing info (vid, trigger, time) from csv to map start and end times per video
        csv_file = v_path + 'camera.csv'
        map_bg = {} # start time
        map_ed = {} # end time
        with open(csv_file) as csvFile:
            readcsv = csv.reader(csvFile)
            for row in readcsv:
                vid = int(row[0])
                trigger = int(row[1])
                tm = float(row[2])
                if (vid + 10) == trigger:
                    map_bg[vid] = tm
                elif (vid+100) == trigger:
                    map_ed[vid] = tm
                else:
                    if vid == -1:
                        if trigger == 0:
                            map_bg[vid] = tm
                        elif trigger == 1:
                            map_ed[vid] = tm
                        else:
                            print('Invalid event.', vid, '-', trigger)
                    else:
                        print('Invalid event.', vid, '-', trigger)

        # Extract trial subclips for all 32 trials of this subject
        for vid in range(32):
            videocilp(v_path+vname, path4save, vid, (map_bg[vid]-map_bg[-1])/1000, (map_ed[vid]-map_bg[-1])/1000)
        print(f"participant {subj} done")

if __name__=='__main__':
    #rootpath = os.path.join('.', 'Raw_data/')
    mainproc('.//Raw_data/')