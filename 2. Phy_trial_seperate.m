% Physiological signal seperation based on trials

rootpath = 'E:/Thesis/Raw_data/';
invalid = [];
for j=1:80 
    file_path = [rootpath num2str(j) '/sub' num2str(j) '/1_raw.edf'];
    if ~exist(file_path, 'file')
        warning('Participant %d file not found! going to next participant', j);
        continue;
    end  
    %% EEG
    file_name = '1_ICA.set';
    EEG = pop_loadset('filename',file_name, 'filepath',[rootpath num2str(j) '/sub' num2str(j) '/']);
    
    % Read annotations from original EDF file
    [~, anns] = edfread([rootpath num2str(j) '/sub' num2str(j) '/1_raw.edf']);
    onsets = seconds(anns.Onset);  % Event times in seconds
    labels = anns.Annotations;     % Trigger labels 
    
    % Convert annotations to numeric triggers
    tris = str2double(labels);
    valid = ~isnan(tris);  % Filter out non-numeric annotations
    tris = tris(valid);
    onsets = onsets(valid);
    
    % Convert time stamps to sample indices
    srate = EEG.srate;  % Get sampling rate from EEG struct
    tri_inds = round(onsets * srate) + 1;  % Convert seconds to samples 
    
    % Trigger processing
    bg = find(tris==60, 1);  % Find first occurrence of trigger 60 (Experiment start marker)
    ed = numel(tris);
    
    tri_inds_t = [];
    tris_t = [];
    for cur = bg:ed-1 % identify pairs with consecutive triggers difference 90 (e.g., 50 and 140 for event start/end)
        if (tris(cur+1)-tris(cur)) == 90
            tri_inds_t = [tri_inds_t, tri_inds(cur), tri_inds(cur+1)];
            tris_t = [tris_t, tris(cur), tris(cur+1)];
        end
    end
    tri_inds = tri_inds_t;
    tris = tris_t;
    
    eeg_datas = [];
    if size(tri_inds, 2) == 68 %Checking if there are 68 triggers
        assert(tris(3)==50, 'Missing expected trigger 50 at position 3')
        for i=3:2:68
            assert((tris(i+1)-tris(i)) == 90)
            eeg_data = EEG.data(:, tri_inds(i):tri_inds(i+1)-1);
            vids = ones(1, size(eeg_data, 2))*(tris(i)-10); %Assign video ID as Trigger-10 (50-10=40 for Rest ID).
            eeg_datas = [eeg_datas, [eeg_data;vids]];
        end
    end
    %% PPG and GSR
    gsr_ = csvread([rootpath num2str(j) '/' num2str(j) '/raw_gsr.csv']);
    gsr_data = gsr_(:,2)'; %2nd column of raw_GSR.csv is the gsr data
    triggers = gsr_(:,3)'; %3rd column of raw_GSR.csv is the trigger data
    
    %similar algorithm for trial seperataion as EEG
    tri_inds = find(triggers > 0);
    tris = triggers(tri_inds);
    bg = find(tris==60);
    ed = size(tris,2);
    tri_inds_t= [];
    tris_t = [];
    for cur = bg:ed-1
       if (tris(cur+1)-tris(cur)) == 90
           tri_inds_t = [tri_inds_t, tri_inds(cur),tri_inds(cur+1)];
           tris_t = [tris_t, tris(cur), tris(cur+1)];
       end
    end
    tri_inds = tri_inds_t;
    tris = tris_t;
    assert(tris(3)==50)
    gsr_datas = [];
    if size(tri_inds, 2)==68
       for i=3:2:68
          assert((tris(i+1)-tris(i)) == 90)
          data_ = gsr_data(:, tri_inds(i):tri_inds(i+1)-1);
          vids = ones(1, size(data_, 2))*(tris(i)-10);
          gsr_datas = [gsr_datas, [data_;vids]];
       end
    end
    
    
    %%PPG
    ppg_ = csvread([rootpath num2str(j) '/' num2str(j) '/raw_ppg.csv']);
    ppg_data = ppg_(:,2)'; %2nd column of raw_ppg.csv is the ppg data
    triggers = ppg_(:,3)'; %3rd column of raw_ppg.csv is the trigger data
    
    %similar algorithm for trial seperataion as EEG
    tri_inds = find(triggers > 0);
    tris = triggers(tri_inds);
    bg = find(tris==60);
    ed = size(tris,2);
    tri_inds_t= [];
    tris_t = [];
    for cur = bg:ed-1
       if (tris(cur+1)-tris(cur)) == 90
           tri_inds_t = [tri_inds_t, tri_inds(cur),tri_inds(cur+1)];
           tris_t = [tris_t, tris(cur), tris(cur+1)];
       end
    end
    tri_inds = tri_inds_t;
    tris = tris_t;
    assert(tris(3)==50)
    ppg_datas = [];
    if size(tri_inds, 2) ==68
       for i=3:2:68
          assert((tris(i+1)-tris(i)) == 90)
          data_ = ppg_data(:, tri_inds(i):tri_inds(i+1)-1);
          vids = ones(1, size(data_, 2))*(tris(i)-10);
          ppg_datas = [ppg_datas, [data_;vids]];
       end
    end
    
    
    if size(gsr_datas, 2)==0 || size(ppg_datas, 2)==0
       invalid = [invalid, j];
    end
    
    save(['E:/Thesis/Aligned_data/' num2str(j) '/Sig_data.mat'], 'eeg_datas', 'gsr_datas', 'ppg_datas');
    fprintf('Participant %d complete\n',j); 

end
