import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import CSVLogger
from keras.layers import LSTM, Dense, Masking, Dropout, Concatenate, Input
from keras.regularizers import l1
from keras.models import Model
from tensorflow.keras.optimizers import Adam

data = sio.loadmat('full_features_affectnet.mat')
feas = data['feas'][:,:]
label = data['label'].ravel()
participant = data['participant'].ravel()
video = data['vids'].ravel()
unique_videos = np.unique(video)
unique_participants = np.unique(participant)
accuracies,conf_matrices, auc_all, f1_all = [], [], [],[]
feature = np.zeros_like(feas, dtype=np.float32)

# Normalize each participant separately
for pid in unique_participants:
    mask = (participant == pid)
    participant_features = feas[mask]
    scaler = StandardScaler()
    feature[mask] = scaler.fit_transform(participant_features)

# Leave one subject out cross validation
for test_participant in unique_participants:
    print(f"\nProcessing participant {test_participant}...")
    # Split data
    train_idx = participant != test_participant
    test_idx = participant == test_participant
    x_train, x_test = feature[train_idx], feature[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]
    train_video, test_video = video[train_idx], video[test_idx]
    train_participant = participant[train_idx]

    eeg_sequences, gsr_sequences, ppg_sequences,face_sequences, train_labels  =[],[],[],[],[]

    for vid in np.unique(train_video):
        for pid in np.unique(train_participant):
            mask = (train_video == vid) & (train_participant == pid)
            eeg_data = x_train[mask, 0:90]
            gsr_data = x_train[mask, 90:130]
            ppg_data = x_train[mask, 130:164]
            face_data = x_train[mask, 164:]
            video_label = y_train[mask][0]
            eeg_sequences.append(eeg_data)
            gsr_sequences.append(gsr_data)
            ppg_sequences.append(ppg_data)
            face_sequences.append(face_data)
            train_labels.append(video_label)

    length = max(seq.shape[0] for seq in eeg_sequences) # Find the maximum length of a sample

    # Padding of all the samples to have uniform length
    eeg_data = pad_sequences(eeg_sequences, maxlen=length, padding='post', dtype='float32', value=-999)
    gsr_data = pad_sequences(gsr_sequences, maxlen=length, padding='post', dtype='float32',value=-999)
    ppg_data = pad_sequences(ppg_sequences, maxlen=length, padding='post', dtype='float32',value=-999)
    face_data = pad_sequences(face_sequences, maxlen= length, padding='post', dtype='float32',value=-999)
    train_label = np.array(train_labels)

    # Train and validate set
    eeg_train, eeg_val, gsr_train, gsr_val, ppg_train, ppg_val, face_train, face_val, label_train, label_val = train_test_split(
        eeg_data, gsr_data, ppg_data, face_data, train_label, test_size=0.015, random_state=42,
        stratify=train_label
    )
    test_sequences, test_labels =[],[]
    for vid in np.unique(test_video):
        mask = test_video == vid
        video_data = x_test[mask]
        video_label = y_test[mask][0]
        test_sequences.append(video_data)
        test_labels.append(video_label)
    test_data = pad_sequences(test_sequences, maxlen= length, padding='post', dtype='float32',value=-999)
    test_label = np.array(test_labels)

    # Split modalities for test data
    eeg_test = test_data[:, :, 0:90]
    gsr_test = test_data[:, :, 90:130]
    ppg_test = test_data[:, :, 130:164]
    face_test = test_data[:, :, 164:]

    # Convert labels to categorical
    num_classes = len(np.unique(label))
    train_label_cat = to_categorical(label_train, num_classes=num_classes)
    valid_label_cat = to_categorical(label_val, num_classes=num_classes)
    test_label_cat = to_categorical(test_label, num_classes=num_classes)

    # Define input shapes
    eeg_shape = (length, eeg_data.shape[2])
    gsr_shape = (length, gsr_data.shape[2])
    ppg_shape = (length, ppg_data.shape[2])
    facial_shape = (length, face_data.shape[2])

    best_accuracy = 0.0
    best_model = None

    for run in range(2):

        # 2 layered LSTM model architecture

        # EEG branch
        eeg_input = Input(shape=eeg_shape, name='eeg_input')
        eeg_masked = Masking(mask_value=-999)(eeg_input)
        eeg_lstm = LSTM(256,kernel_regularizer=l1(l1=0.005), return_sequences=True )(eeg_masked)
        eeg_lstm = Dropout(0.1)(eeg_lstm)
        eeg_lstm = LSTM(128, kernel_regularizer=l1(l1=0.005))(eeg_lstm)

        # GSR branch
        gsr_input = Input(shape=gsr_shape, name='gsr_input')
        gsr_masked = Masking(mask_value=-999)(gsr_input)
        gsr_lstm = LSTM(256,kernel_regularizer=l1(l1=0.005), return_sequences=True)(gsr_masked)
        gsr_lstm = Dropout(0.1)(gsr_lstm)
        gsr_lstm = LSTM(128, kernel_regularizer=l1(l1=0.005))(gsr_lstm)

        # PPG branch
        ppg_input = Input(shape=ppg_shape, name='ppg_input')
        ppg_masked = Masking(mask_value=-999)(ppg_input)
        ppg_lstm = LSTM(256,kernel_regularizer=l1(l1=0.005), return_sequences=True)(ppg_masked)
        ppg_lstm = Dropout(0.1)(ppg_lstm)
        ppg_lstm = LSTM(128, kernel_regularizer=l1(l1=0.005))(ppg_lstm)

        # Facial branch
        facial_input = Input(shape=facial_shape, name='facial_input')
        facial_masked = Masking(mask_value=-999)(facial_input)
        facial_lstm = LSTM(256,kernel_regularizer=l1(l1=0.005), return_sequences= True)(facial_masked)
        facial_lstm =Dropout(0.1)(facial_lstm)
        facial_lstm = LSTM(128, kernel_regularizer=l1(l1=0.005))(facial_lstm)

        # Concatenate all modalities (Feature level fusion)
        merged = Concatenate()([eeg_lstm, gsr_lstm, ppg_lstm, facial_lstm])

        dense = Dense(512, activation='relu')(merged)
        output = Dense(num_classes, activation='softmax')(dense)

        model = Model(inputs=[eeg_input, gsr_input, ppg_input, facial_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate= 0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

        # Create the CSVLogger callback to store per epoch results
        csv_logger = CSVLogger(f'./lstm_affectnet/all features/log_participant_{int(test_participant)}.log', append=True)

        # Train the model
        model.fit(
            [eeg_train, gsr_train, ppg_train, face_train], train_label_cat,
            epochs=50,
            batch_size=32,
            validation_data=([eeg_val, gsr_val, ppg_val, face_val], valid_label_cat),
            callbacks=[csv_logger],
            verbose=1
        )
        # Evaluate model with metrics
        loss, acc = model.evaluate([eeg_test, gsr_test, ppg_test, face_test], test_label_cat, verbose=0)
        print(f'participant {test_participant}, run{run} : test accuracy {acc}')

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            y_pred_probs = best_model.predict([eeg_test, gsr_test, ppg_test, face_test])
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(test_label_cat, axis=1)
            f1 = f1_score(y_true, y_pred, average='macro')
            auc = roc_auc_score(y_true, y_pred_probs,
                                multi_class='ovo')  # Multiclass 'ovo' to average AUC of all possible pairwise combinations of classes
            # Compute confusion matrix for this participant
            cm = confusion_matrix(y_true, y_pred, labels=range(num_classes), normalize='true')

    conf_matrices.append(cm)
    accuracies.append(best_accuracy)
    auc_all.append(auc)
    f1_all.append(f1)
    print(f"Test accuracy for participant {test_participant}: {best_accuracy:.4f}")
    print(f"Test AUC for participant {test_participant}: {auc:.4f}")
    print(f"F1 Score (macro) for participant {test_participant}: {f1:.4f}")
    print(f'predicted labels for participant {test_participant}: {y_pred}')

    # save the model to be used later for explainability analysis
    best_model.save(f'./lstm_affectnet/all features/participant_{int(test_participant)}.h5')

    # Store results
    result = {
        'participant': test_participant,
        'metrics': {'test_accuracy': best_accuracy,
                    'auc': auc,
                    'f1_score': f1,
                    'confusion_matrix': cm,
                    'predictions': y_pred}
    }
    # Save this fold's results to disk
    joblib.dump(result, f'./lstm_affectnet/all features/metrics_participant_{int(test_participant)}.pkl')

print(f"\nMean accuracy across participants: {np.mean(accuracies):.4f}")
print(f"\nMean AUC across participants: {np.mean(auc_all):.4f}")
print(f"\nMean F1 score across participants: {np.mean(f1_all):.4f}")

# Compute and display the mean confusion matrix across all participants
cm = np.mean(conf_matrices, axis=0)
figure, ax = plt.subplots(dpi=200)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative", "Mixed"])
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Mean Confusion Matrix", fontweight='bold')
ax.set_xlabel("Predicted Label", fontweight='bold')
ax.set_ylabel("True Label", fontweight='bold')
plt.tight_layout()
plt.close()