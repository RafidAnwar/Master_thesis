% preprocessing script for raw eeg data

file_root={'E:/Thesis/Raw_data/'};
% Start parallel pool 
if isempty(gcp('nocreate'))
    parpool('local'); % Uses available CPU cores
end
parfor j=1:80 %parallel for loop
    file_path = [char(file_root(1)) num2str(j) '/sub' num2str(j) '/1_raw.edf'];
    if ~exist(file_path, 'file')
        warning('Participant %d file not found! going to next participant', j);
        continue;
    end
    EEG = pop_biosig(file_path); %using biosig toolbox load raw EEG Data from EDF file
    
    % channel rename
    for i = 1:numel(EEG.chanlocs)
        idx =   strfind(EEG.chanlocs(i).labels,'-');
        if ~isempty(idx) %For labels like "EEG FP1-Ref" change to FP1
            tmp = strsplit(EEG.chanlocs(i).labels(1:idx-1),' ');
            EEG.chanlocs(i).labels = tmp{1,2};
        else %For labels like "EEG O1" change to O1
            EEG.chanlocs(i).labels = EEG.chanlocs(i).labels(5:7)
        end
    end
    
    EEG = pop_chanedit(EEG, 'lookup','E:/Thesis/eeglab2025.0.0/plugins/dipfit/standard_BEM/elec/standard_1005.elc'); % Load international 10-20 system 
    EEG = pop_select( EEG,'nochannel',{'A1','A2','ger', 'X1', 'X2', 'X3'}); %Remove selected channels
    EEG = pop_reref( EEG, 9);  %Rereference to CM electrode
    EEG = pop_eegfiltnew(EEG, 'locutoff',1); % High-pass filter at 1Hz
    EEG = pop_eegfiltnew(EEG, 'hicutoff',50); % Low-pass filter at 50Hz
    EEG = pop_eegfiltnew(EEG, 'locutoff',49,'hicutoff',51,'revfilt',1); % Notch filter at 49-51Hz
    EEG = pop_rmbase( EEG, [],[]); % Remove baseline
    EEG = pop_eegthresh(EEG, 1, 1:EEG.nbchan, -100, 100, EEG.xmin, EEG.xmax, 1, 0); % Threshold rejection (±100microV)
    
    % Wavelet denoising:
    fprintf('Applying wavelet EMG suppression for %d\n',j);
    wname = 'sym4'; %Symlet-4 for EMG Sepration
    level = 4;  % Decomposition level for 300 hz sampling rate

    for ch = 1:EEG.nbchan
        [C, L] = wavedec(EEG.data(ch,:), level, wname); %Wavelet decomposition of original EEG
        % Only zero out D1 (EMG >75Hz), leave D2-D4 (0-75Hz)
        k = 1  % Target only D1 (75-150 Hz)
        C = wthcoef('d', C, L, k, 's', 'rigrsure'); %threshold targetting
        EEG.data(ch,:) = waverec(C, L, wname); %Reconstruction to final EEG
    end
    
    %Indepandant component analysis
    fprintf('Applying ICA for %d\n',j);
    EEG = pop_runica(EEG, 'extended',1,'interupt','on'); % ICA
    EEG = iclabel(EEG);
	noiselabel = round(EEG.etc.ic_classification.ICLabel.classifications(:,:)*100)
	noisethreshold = [0 0; 0.9 1;0.9 1;0 0;0 0;0 0;0 0]; %Muscle and Eye category confidence threshold (0.9-1)
	EEG = pop_icflag(EEG, noisethreshold);
	EEG = pop_subcomp(EEG, []); 
    EEG = pop_saveset( EEG, 'filename',  '1_ICA.set','filepath', [char(file_root(1)) num2str(j) '/sub' num2str(j)]); %Save data
    fprintf('Participant %d complete\n',j);
end
% Cleanup parallel pool
delete(gcp);