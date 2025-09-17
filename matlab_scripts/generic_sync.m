%% initialize workspace and paths
clear; close all;clc;
tic;

% add path to custom FFmpeg functions (used later for video/audio processing)
addpath('/Users/agusamilibia/Documents/MATLAB/ffmpeg-r8');

% set base paths
path_to_data = '/Users/agusamilibia/Documents/capstone-data/Jamail'; %<----CHANGE HERE THE PATH
date = '2025-04-08';%<----CHANGE HERE THE DATE
path_to_processed = fullfile(path_to_data, date, 'Processed');
disp(path_to_processed);


% create processed data folder if it doesn't exist
if ~isfolder(path_to_processed)
    mkdir(path_to_processed);
end

% find relevant files in the raw data directory
raw_folder = fullfile(path_to_data, date, 'Raw');

audio_files = dir(fullfile(raw_folder, '*.wav'));
disp("Audio files found:");
disp({audio_files.name});

video_files = dir(fullfile(raw_folder, '*.mp4'));
disp("Video files found:");
disp({video_files.name});
% define IDs of FLIR cameras used in the session
camera_ids = [];

for i = 1:length(video_files)
    tokens = regexp(video_files(i).name, '\.(\d+)\.mp4$', 'tokens', 'once');
    if ~isempty(tokens)
        id = str2double(tokens{1});
        if ~ismember(id, camera_ids)
            camera_ids(end+1) = id;
        end
    end
end

json_files = dir(fullfile(raw_folder, '*.json'));
disp("JSON files found:");
disp({json_files.name});

%% Load and Plot Audio, extract binary signal


for i = 1:length(audio_files)
    
    % Load current audio file
    current_audio_file = fullfile(raw_folder, audio_files(i).name);
    [y(i,:), fs(i,:)] = audioread(current_audio_file); % y: signal, fs: sampling frequency

    % Create time vector for plotting
    audio_t = linspace(0, length(y(i,:)) / fs(i,:), length(y(i,:))); % Time vector in seconds

    % Normalize and convert Channel 3 to binary signal
    if i == 3 
        y(i,:) = (y(i,:) - min(y(i,:))) / (max(y(i,:)) - min(y(i,:))); % Normalize to [0,1]
        binary_signal = y(i,:) > 0.5; % Convert to logical signal (0 or 1)
    end

    % Plot audio signal
    figure(1);
    ax(i) = subplot(3,1,i);
    plot(audio_t, y(i,:));
    title(['Audio (Channel ' int2str(i) ')']);
    xlabel('Time (s)');
    ylabel('Amplitude');
    linkaxes(ax);
    grid on;
end

binary_signal = flip(binary_signal);

figure(20);
plot(1:length(binary_signal), binary_signal, 'r');  % Plot by sample index
xlabel('Sample Index');
ylabel('Binary Signal');
title('Binary Signal (vs Sample Index)');
ylim([-0.1, 1.1]);
xlim([75000,76000]); 

%% Extract Serial Frame ID from Audio Channel 3

timestamp = zeros(128,1);              
audio_serial_index = zeros(128,1); % stores the sample index in the audio signal for each sync start
clipped_signal = zeros(128,231);  % stores 230-sample windows of the binary signal (each contains a full sync block)
clipped_signal_byte = [];      % will store the extracted bytes (8-bit sequences) from each clipped signal

% Define positions of byte segments in the 230-sample sync block
transition_points = 6:47:230;          % Each new byte starts every 47 samples
offset_from_transition = [4,9,14,19,23,28,33,37];  % Offsets to extract 8 bits from each transition point

current_ind = 1;   %for scanning the binary signal
count = 1;   % for sync events found

% Loop through the binary signal to extract sync frames

% going backwards
while current_ind < length(binary_signal)
    current_value = binary_signal(current_ind);
    if current_value == 1
        current_ind = current_ind + 1;% skip if still in the high signal 
        continue;
    else

        timestamp(count) = audio_t(current_ind);  % Store timestamp at start of sync
        audio_serial_index(count) = current_ind; % Store index in the audio signal
        if current_ind + 231 > length(binary_signal)
            break;  
        end
        clipped_signal(count,:) = flip(binary_signal(current_ind:current_ind+230));  % Extract full 230-sample window
        current_ind = current_ind + 1100;                         % Move index forward by one sync block size

        % Extract 8-bit sequences for each of the 5 serial bytes
        clipped_signal_byte(1,count,:) = clipped_signal(count, transition_points(1)+offset_from_transition);
        clipped_signal_byte(2,count,:) = clipped_signal(count, transition_points(2)+offset_from_transition);
        clipped_signal_byte(3,count,:) = clipped_signal(count, transition_points(3)+offset_from_transition);
        clipped_signal_byte(4,count,:) = clipped_signal(count, transition_points(4)+offset_from_transition);
        clipped_signal_byte(5,count,:) = clipped_signal(count, transition_points(5)+offset_from_transition);

        % %Optional: visualize the clipped signal (serial transmision)
        % figure(3);  
        % plot(clipped_signal(count,:)); ylim([-.1,1.1]); xlim([-100,580]);
        % pause(.05);
        % title('Serial transmission');

        count = count + 1;  % Move to next sync block
    end

    % Display progress in console as a fraction of the total signal
    disp(current_ind / length(binary_signal))
end

% Convert extracted binary bits to string representations, one per byte
byte_string1 = join(string(flip(squeeze(clipped_signal_byte(1,:,1:end-1)),2)),'',2);
byte_string2 = join(string(flip(squeeze(clipped_signal_byte(2,:,1:end-1)),2)),'',2);
byte_string3 = join(string(flip(squeeze(clipped_signal_byte(3,:,1:end-1)),2)),'',2);
byte_string4 = join(string(flip(squeeze(clipped_signal_byte(4,:,1:end-1)),2)),'',2);
byte_string5 = join(string(flip(squeeze(clipped_signal_byte(5,:,1:end-1)),2)),'',2);

% Concatenate all 5 bytes and convert to decimal frame IDs
serial_id = flip(bin2dec(strcat(byte_string5,byte_string4,byte_string3,byte_string2,byte_string1)));
timestamp = timestamp(1:length(serial_id));



% Optional: scatter plot of frame ID over time
figure(4); 
scatter(timestamp, serial_id);
title('Serial Frame ID Time in Audio Recording') 
ylim([12259716,12262189]);
xlabel('Time in Audio Recording (Seconds)')
ylabel('Serial Frame ID')



%% Organize Video Files
for i = 1:length(video_files)
    % extract metadata from filename: TestVideo<id>_<date>_<time>.<cameraID>.mp4
    tmp_file_name_scaned = sscanf(video_files(i).name,'video_%d_%d.%d.mp4'); %<--- CHANGE THE NAME HERE
    % extract camera ID from filename (last value in the pattern)
    video_file_camera_id(i) = tmp_file_name_scaned(end);
    % extract time or sequence value from filename (second-to-last value)
    video_file_time(i) = tmp_file_name_scaned(end-1);
end

% organize videos by camera and recording time (cameras x videos for each camera)
% create a cell array with dimensions (num_cameras x num_recording_times)
unique_recordings = unique(video_file_time); %counting how many dif videos i have for each camera
gouped_video_names_by_camera = cell(length(camera_ids), length(unique_recordings));


%array where each row is a camera, each column a time
for i = 1:length(video_files)
    time_ind = find(unique_recordings == video_file_time(i)); %time = column index
    cam_ind = find(camera_ids == video_file_camera_id(i));%camera= row index
    % store the video file name in the corresponding cell
    gouped_video_names_by_camera{cam_ind, time_ind} = video_files(i).name;
end


%% Matching Audio with JSON Metadata
for i = 1:length(json_files)
    tokens = regexp(json_files(i).name, 'video_(\d+)_(\d+)', 'tokens'); %<--- CHANGE THE NAME HERE

    if isempty(tokens)
        error(['Filename does not match expected pattern: ' json_files(i).name]);
    end

    tmp_file_name_scaned = str2double(tokens{1});
    json_file_order(i) = tmp_file_name_scaned(end);

    % Load JSON file content
    fileName = [json_files(i).folder '/' json_files(i).name];
    str = fileread(fileName); % dedicated for reading files as text
    json_data{i} = jsondecode(str);
    frame_rate(i) = 1 / mean(diff(json_data{i}.timestamps(:,1)) * 1e-9);
    %estimate video duration based on timestamps
    video_duration(i) = mean(diff(json_data{i}.timestamps(:,1)) * 1e-9) * length(json_data{i}.timestamps(:,1));

   
    % fix initial -1 values in chunk_serial_data
    video_serial = json_data{i}.chunk_serial_data(:,1);     % serial IDs
    first_valid_idx = find(video_serial ~= -1, 1, 'first'); % index of first usable ID
    estimated_first_val = video_serial(first_valid_idx) - (first_valid_idx - 1); % calculate missing initial value
    video_serial(1:first_valid_idx-1) = estimated_first_val:(video_serial(first_valid_idx)-1); 

    %match video frames with the segment in the audio
    audio_clip = find_audio_subset(y, serial_id, audio_serial_index, ...
                                   video_serial, ...
                                   json_data{i}.timestamps(:,1));

    %save extracted audio 
    for k = 1:height(audio_clip)
        audio_folder_path = fullfile(path_to_processed, 'Audio', num2str(k));
        if ~isfolder(audio_folder_path)
            mkdir(audio_folder_path);
        end
        audio_file_name{k} = [audio_folder_path 'Audio_' num2str(tmp_file_name_scaned(2)) '_' ...
                              num2str(json_file_order(i)) '_' num2str(k) '.wav'];
        audiowrite(audio_file_name{k}, audio_clip(k,:), fs(k));
    end

    %% Aligning audio and video
    %process and align video for each camera
    for j = 1:height(gouped_video_names_by_camera)
        video_folder_path = fullfile(path_to_processed, 'Video', num2str(j));
        if ~isfolder(video_folder_path)
            mkdir(video_folder_path);
        end

        input_video_name = fullfile(raw_folder, gouped_video_names_by_camera{j,i});
        video_file_name_short = gouped_video_names_by_camera{j,i}(1:end-4);
        camera_id = extractAfter(video_file_name_short, '.');

        output_video_name = fullfile(video_folder_path, [video_file_name_short '_corrected.mp4']);
        ffmpeg_command_frame_rate_correction = ['-r ' num2str(frame_rate(i),16) ...
                                                ' -i "' input_video_name '"' ...
                                                ' -c:v mpeg4 -q:v 1 "' output_video_name '"'];
        ffmpegexec(ffmpeg_command_frame_rate_correction);

        final_video_output_name = fullfile(video_folder_path, ...
            ['Video_' num2str(tmp_file_name_scaned(2)) '_' ...
             num2str(tmp_file_name_scaned(end)) '_' camera_id '.mp4']);
        ffmpeg_command_map_audio = ['-i "' output_video_name '" -i "' audio_file_name{1} ...
                                    '" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "' ...
                                    final_video_output_name '"'];
        ffmpegexec(ffmpeg_command_map_audio);
        delete(output_video_name)
    end

end
time = toc;
fprintf("Execution time: %.4f minutes\n", time/60);