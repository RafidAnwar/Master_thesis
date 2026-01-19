import scipy.io as sio
from keras.utils import pad_sequences
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

data = sio.loadmat('./full_features_affectnet.mat')
feas = data['feas'][:, :]
label = data['label'].ravel()
participant = data['participant'].ravel()
video = data['vids'].ravel()
unique_videos = np.unique(video)
unique_participants = np.unique(participant)
feature = np.zeros_like(feas, dtype=np.float32)
preds_mc, preds_all_mc, preds_tta, preds_all_tta = [], [], [],[]
effects_all,accuracies_all = [0, 0, 0, 0], [0, 0, 0, 0]
final_acc = 0


def uncertainty_plots(mc_preds_all, name):  # All plots for explainability using UQ
    avg_fold_uncertainties = []
    avg_uncertainty_misclassified = []
    avg_uncertainty_correct = []
    fold_accuracies = []
    per_class_uncertainty_totals = np.zeros(3)
    per_class_counts = np.zeros(3)
    mean_preds_all, std_preds_all = [], []
    for fold in range(66):

        preds_mc = mc_preds_all[fold]
        true_labels = test_label

        mean_preds = np.mean(preds_mc, axis=0)  # Calculate the mean prediction values for this participant
        std_preds = np.std(preds_mc, axis=0)  # Calculate the uncertainty for this participant by the std
        mean_preds_all.append(mean_preds)
        std_preds_all.append(std_preds)

        pred_classes = np.argmax(mean_preds, axis=1)
        pred_uncertainty = std_preds[np.arange(32), pred_classes]

        # Accuracy for this participant
        acc = accuracy_score(true_labels, pred_classes)
        fold_accuracies.append(acc)

        # Separate uncertainty for correct and misclassified
        correct_mask = (pred_classes == true_labels)
        misclassified_mask = ~correct_mask

        avg_fold_uncertainties.append(np.mean(pred_uncertainty))
        avg_uncertainty_correct.append(np.mean(pred_uncertainty[correct_mask]) if np.any(correct_mask) else 0)
        avg_uncertainty_misclassified.append(
            np.mean(pred_uncertainty[misclassified_mask]) if np.any(misclassified_mask) else 0)

        # Accumulate per-class uncertainty over all participants
        for c in range(3):
            class_mask = (pred_classes == c)
            per_class_uncertainty_totals[c] += np.sum(pred_uncertainty[class_mask])
            per_class_counts[c] += np.sum(class_mask)

    # Average per-class uncertainty across all participants
    avg_per_class_uncertainty = per_class_uncertainty_totals / np.maximum(per_class_counts, 1)

    avg_fold_uncertainties = np.array(avg_fold_uncertainties)
    avg_uncertainty_correct = np.array(avg_uncertainty_correct)
    avg_uncertainty_misclassified = np.array(avg_uncertainty_misclassified)
    mean_preds_all = np.array(mean_preds_all)
    std_preds_all = np.array(std_preds_all)

    # Compute mean and std across participants for each sample and class
    mean_per_sample = np.mean(mean_preds_all, axis=0)
    std_per_sample = np.sqrt(np.mean(std_preds_all ** 2, axis=0))

    num_samples = mean_per_sample.shape[0]
    num_classes = mean_per_sample.shape[1]
    x = np.arange(num_samples)

    # Error plot for showing mean probabilities with uncertainties as error bar
    plt.figure(figsize=(16, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Custom colors
    class_names = ['Positive', 'Negative', 'Mixed']
    for c in range(num_classes):
        label_text = f'{class_names[c]}'
        plt.errorbar(x, mean_per_sample[:, c], yerr=std_per_sample[:, c],
                     label=label_text, alpha=0.85, capsize=5, fmt='-o',
                     markersize=8, elinewidth=2,
                     color=colors[c % len(colors)], markerfacecolor='white', markeredgewidth=2)

    plt.xlabel("Sample Index", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Prediction Probability", fontsize=14, fontweight='bold')
    plt.title(f"Mean Predictions ± Uncertainty Across Participants for Each Sample during {name}",
              fontsize=16, weight='bold')
    plt.legend(title='Classes', title_fontsize='13', fontsize='11', loc='best', frameon=True)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xticks(np.arange(0, 32, step=1), fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{name}_mean.png", dpi=600)
    plt.show()

    # per participant uncertainty plot
    plt.figure(figsize=(18, 5))
    plt.bar(range(66), avg_fold_uncertainties)
    plt.xlabel("Participant Index", fontsize=12, fontweight='bold')
    plt.xticks(np.arange(0, 66, step=1), fontsize=7, fontweight='bold')
    plt.ylabel("Average Prediction Uncertainty", fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(f"Average {name} Uncertainty per Participant", fontsize=12, fontweight='bold')
    mean_value = np.mean(avg_fold_uncertainties)
    plt.axhline(mean_value, linestyle='--', linewidth=2, label=f"Mean = {mean_value:.3f}")
    plt.tight_layout()
    plt.savefig(f"{name}_participant.png", dpi=600)
    plt.show()

    # Bar plot of avg uncertainty on correct vs misclassified samples per participant (mean value)
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    plt.bar(['Correct', 'Misclassified'],
            [np.mean(avg_uncertainty_correct), np.mean(avg_uncertainty_misclassified)],
            color=['seagreen', 'red'])
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.ylabel("Average Prediction Uncertainty", fontsize=12, fontweight='bold')
    plt.title(f"Average {name} Uncertainty: Correct vs Misclassified Predictions", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{name}_corr.png", dpi=600)
    plt.show()

    # Average per-class uncertainty
    plt.figure(figsize=(8, 6))
    sns.barplot(x=np.arange(3), y=avg_per_class_uncertainty, palette=colors)
    plt.ylabel("Average Prediction Uncertainty", fontsize=12, fontweight='bold')
    plt.xticks(np.arange(3), ['Positive', 'Negative', 'Mixed'], fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(f"Average {name} Uncertainty per Class", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{name}_class.png", dpi=600)
    plt.show()

    # reliability diagram
    def reliability_diagram(mean_preds, true_labels, n_bins=10, name=name):

        predicted_confidence = np.max(mean_preds, axis=1)
        predicted_classes = np.argmax(mean_preds, axis=1)

        # Bin the predictions
        bins = np.linspace(0.5, 1., n_bins + 1)
        binids = np.digitize(predicted_confidence, bins) - 1

        accuracies = []
        confidences = []
        for i in range(n_bins):
            idx = binids == i
            if np.sum(idx) > 0:
                acc = np.mean(predicted_classes[idx] == true_labels[idx])
                conf = np.mean(predicted_confidence[idx])
                accuracies.append(acc)
                confidences.append(conf)
            else:
                accuracies.append(np.nan)
                confidences.append((bins[i] + bins[i + 1]) / 2)

        plt.figure(figsize=(8, 6))
        plt.plot(confidences, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=3, markersize=10,
                 label='Observed')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Predicted Confidence', fontsize=12, fontweight='bold')
        plt.ylabel('Observed Accuracy', fontsize=12, fontweight='bold')
        plt.xlim([0.3, 1])
        plt.ylim([0.3, 1])
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.title(f'Average {name} Reliability Diagram', fontsize=12, fontweight='bold')
        plt.legend(frameon=True, fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{name}_reliability.png", dpi=600)
        plt.show()

    # Run reliability diagram for all participants pooled
    all_mean_preds = []
    all_true_labels = []
    for fold in range(66):
        preds_mc = mc_preds_all[fold]
        true_labels = test_label
        mean_preds = np.mean(preds_mc, axis=0)  # (32, 3)

        all_mean_preds.append(mean_preds)
        all_true_labels.append(true_labels)

    all_mean_preds = np.concatenate(all_mean_preds, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    reliability_diagram(all_mean_preds, all_true_labels, n_bins=10)
# Normalize each participant separately
for pid in unique_participants:
    mask = (participant == pid)
    participant_features = feas[mask]
    scaler = StandardScaler()
    feature[mask] = scaler.fit_transform(participant_features)

for test_participant in [1,2,5,8,9,10,11,12,14,15,18,19,20,21,22,23,24,25,26,28,29,30,32,33,34,35,
                36,37,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,60,62,
                63,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80]:
    print(f"\nProcessing participant {test_participant}...")
    # Split data
    train_idx = participant != test_participant
    test_idx = participant == test_participant
    x_train, x_test = feature[train_idx], feature[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]
    train_video, test_video = video[train_idx], video[test_idx]

    test_sequences, test_labels = [], []
    for vid in np.unique(test_video):
        mask = test_video == vid
        video_data = x_test[mask]
        video_label = y_test[mask][0]
        test_sequences.append(video_data)
        test_labels.append(video_label)
    test_data = pad_sequences(test_sequences, maxlen=29, padding='post', dtype='float32')
    test_label = np.array(test_labels)

    # Split modalities for test data
    eeg_test = test_data[:, :, 0:90]
    gsr_test = test_data[:, :, 90:130]
    ppg_test = test_data[:, :, 130:164]
    face_test = test_data[:, :, 164:]

    # Load the saved 2 layered LSTM model for each participant
    file = f'./lstm_affectnet/all features/participant_{int(test_participant)}.h5'
    model = load_model(file)

    def predict_with_MC(model, inputs, n_iter=50): # Monte carlo dropout UQ

        preds = []
        for _ in range(n_iter):
            # call model with training=True to keep dropout active
            pred = model(inputs, training=True)
            preds.append(pred.numpy())

        preds = np.array(preds)
        return preds

    def magnitude_warping(modality_data, sigma=0.2, knot=4): # Augment the data using magnitude warping

        warped_data = np.empty_like(modality_data)
        time_steps = modality_data.shape[1]

        # Define knot positions evenly spaced across time dimension
        knot_x = np.linspace(0, time_steps - 1, knot + 2)

        for i in range(modality_data.shape[0]):  # for each sample
            # Generate random scaling factors for spline control points near 1
            random_scaling = np.random.normal(loc=1.0, scale=sigma, size=knot_x.shape)
            # Create cubic spline from knot_x points and random scalings
            spline = CubicSpline(knot_x, random_scaling, bc_type='clamped')

            # Compute smooth scaling curve for all time points
            scaling_curve = spline(np.arange(time_steps))
            scaling_curve = scaling_curve.reshape(-1, 1)

            # Apply scaling curve to all features for this sample
            warped_data[i] = modality_data[i] * scaling_curve

        return warped_data

    def predict_with_tta(model, inputs, n_augment=50): # Test time augmentation UQ

        preds_all = []

        for i in range(n_augment):
            augmented_inputs = []
            for modality_data in inputs:
                # call the magnitude warping to augment per modality data
                augmented_modality = magnitude_warping(modality_data, sigma=0.2, knot=4)
                augmented_inputs.append(augmented_modality)

            preds_mc = model.predict(augmented_inputs, verbose=0)
            preds_all.append(preds_mc)

        preds_all = np.array(preds_all)

        return preds_all

    def compute_perturbation_effect_batch(model, inputs): # Perturbation method to determine modality influence

        orig_preds = model.predict(inputs)
        base_acc = accuracy_score(test_label, np.argmax(orig_preds, axis=1))
        effects, modality_accuracies = [], []
        for i in range(len(inputs)):
            perturbed_inputs = [arr.copy() for arr in inputs]
            amp = np.max(np.abs(perturbed_inputs[i]))
            # add a random noise of value 10% of the maximum amplitude to the modality data
            noise = np.random.normal(0, 0.1 * amp, perturbed_inputs[i].shape)
            perturbed_inputs[i] += noise

            perturbed_preds = model.predict(perturbed_inputs)
            acc = accuracy_score(test_label, np.argmax(perturbed_preds, axis=1))
            # Use mean difference over all samples; L2 norm per sample then mean
            diff_per_sample = np.linalg.norm(orig_preds - perturbed_preds, axis=1)
            mean_diff = np.mean(diff_per_sample)
            effects.append(mean_diff)
            modality_accuracies.append(acc)

        return effects, orig_preds, base_acc, modality_accuracies
    # Usage:
    inputs = [eeg_test, gsr_test, ppg_test, face_test]

    effects, original_predictions, base_acc, modality_accuracies = compute_perturbation_effect_batch(model, inputs)
    final_acc = final_acc + base_acc
    for i in range(4):
        effects_all[i] += effects[i]
        accuracies_all[i] += modality_accuracies[i]

    preds = predict_with_MC(model, inputs)
    preds_mc.append(preds)

    preds_all = predict_with_tta(model, inputs, n_augment=50)
    preds_tta.append(preds_all)

# Modality importance calculation
print(f'Final accuracy = {final_acc / 66:.4f}')
for modality_name, effect in zip(['EEG', 'GSR', 'PPG', 'Face'], effects_all):
    print(f"Perturbing {modality_name} for all samples changed prediction by {effect / 66:.4f}")

for mod_name, acc in zip(['EEG', 'GSR', 'PPG', 'Face'], accuracies_all):
    print(f"Accuracy after perturbing {mod_name}: {acc / 66:.4f} (drop: {(final_acc - acc) / 66:.4f})")

# Uncertainty quantification
preds_all_mc = np.stack(preds_mc, axis=0)
preds_all_tta = np.stack(preds_tta, axis=0)



uncertainty_plots(preds_all_mc, name='Monte Carlo Dropout')
uncertainty_plots(preds_all_tta, name= 'Test Time Augmentation')
