import cv2
import os
import numpy as np
import scipy.io as sio
import mediapipe as mp
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import img_to_array

# Function to preprocess the sample frame according to Emo-affectnet model requirement
def pre_processing(img):

    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = img[..., ::-1] # Convert RGB to BGR

    # Mean subtraction (channel-wise)
    img[..., 0] -= 91.4953  # Blue
    img[..., 1] -= 103.8827 # Green
    img[..., 2] -= 131.0912 # Red

    img = np.expand_dims(img, axis=0) # Add batch dimension

    return img

# Function for MediaPipe frame processing and feature extraction
def emo_affectnet(vf):

    model = load_model('./emo_affectnet_model.h5') # Load the pretrained emo-affectnet model
    feature_model = Model(inputs=model.input, outputs=model.get_layer('dense_4').output) # Select last layer before the softmax activation layer as output
    border_indices = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172,
                      136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251,
                      284, 332, 297, 338] # Border indices of the landmarks for the face region

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    video = cv2.VideoCapture(vf)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    features = None
    for sec in range(int(duration)):
        video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = video.read()
        if not success:
            break

        # Process frame with MediaPipe
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        if results.multi_face_landmarks: # crop 3d face mesh using the 468 landmarks
            face_landmarks = results.multi_face_landmarks[0]

            # Get border landmarks in pixel coordinates
            h, w = frame.shape[:2]
            border_points = []
            for idx in border_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                border_points.append([x, y])

            # Create mask from convex hull of border points
            mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(np.array(border_points))
            cv2.fillConvexPoly(mask, hull, 255)

            # Get bounding rect of the convex hull
            x, y, w_crop, h_crop = cv2.boundingRect(hull)

            # Apply mask and crop face mesh
            masked_face = cv2.bitwise_and(frame, frame, mask=mask)
            face = masked_face[y:y + h_crop, x:x + w_crop]

            # pre-process the image so that it can be fed to the emo-affectnet model
            face = pre_processing(face)

            # Extract feature using the emo-affectnet models last dense layer
            feat = feature_model.predict(face, verbose=0)
            if features is None:
                features = feat
            else:
                features = np.vstack((features, feat))

        # If mp fails to detect landmark just detect the face rectangle and crop the facial area
        else:
            h, w = frame.shape[:2]
            results = mp.solutions.face_detection.FaceDetection().process(frame)
            if results.detections: # Crop the frame using the detected face region coordinates
                b = results.detections[0].location_data.relative_bounding_box
                x, y, x2, y2 = int(b.xmin * w), int(b.ymin * h), int((b.xmin + b.width) * w), int(
                    (b.ymin + b.height) * h)
                face = frame[y:y2, x:x2]

                # pre-process the image so that it can be fed to the emo-affectnet model
                face = pre_processing(face)

                # Extract feature using the emo-affectnet models last dense layer
                feat = feature_model.predict(face, verbose=0)
                if features is None:
                    features = feat
                else:
                    features = np.vstack((features, feat))

    return features


def ext_feature(rootpath):
    for sub in [1,2,5,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,28,29,30,32,33,34,35,
                36,37,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,60,62,
                63,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80]:
        path = rootpath + str(sub) + '/'
        vids = []
        emo_class = []
        feas_all = None
        for i in range(32):
            vf = path + str(i) + '.mp4'

            # Call function to for video feature extraction by means of transfer learning
            feas = emo_affectnet(vf)

            if feas_all is None:
                feas_all = feas
            else:
                feas_all = np.vstack((feas_all, feas))

            # Add video emotion class : 0 = positive, 1 = Negative, 2 = Mixed
            if i <= 7:
                emo_class = np.append(emo_class, np.ones(feas.shape[0], np.int32)*0)
            elif 7 < i <= 15:
                emo_class = np.append(emo_class, np.ones(feas.shape[0], np.int32) * 1)
            else:
                emo_class = np.append(emo_class, np.ones(feas.shape[0], np.int32) * 2)

            vids = np.append(vids, np.ones(feas.shape[0], np.int32) * i)
            print(f'video {i} done')

        print(f'participant {sub} done')
        savepath = './Aligned_data/'+ str(sub) + '/'

        if not os.path.exists(savepath):
            os.mkdir(savepath)
        sio.savemat(savepath + 'videoFea_affectnet.mat', {'feas_all': feas_all, 'vids': vids, 'emo_class': emo_class})


if __name__ == '__main__':
    ext_feature('./Aligned_data/')