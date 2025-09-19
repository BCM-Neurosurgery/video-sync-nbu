function audio_clip = find_audio_subset(audio_data, audio_serial, audio_ind, video_serial,timestamps)
%FIND_AUDIO_SUBSET Summary of this function goes here
%   Detailed explanation goes here

video_start_frame_id = video_serial(1);
video_end_frame_id = video_serial(end);
audio_start_frame_id = audio_serial(1);
audio_end_frame_id = audio_serial(end);


ind_of_audio_serial_video_start_match = find(audio_serial == video_start_frame_id);
ind_of_audio_serial_video_end_match = find(audio_serial == video_end_frame_id);

audio_start_ind = audio_ind(ind_of_audio_serial_video_start_match);
audio_end_ind = audio_ind(ind_of_audio_serial_video_end_match+1)-1; %add one to get the audio of the last frame since signal is at start of frame
start_padd = [];
end_padd = [];
if isempty(audio_start_ind)
    audio_start_ind = 1;
    frame_audio_starts = find(video_serial == audio_start_frame_id);
    duration_before_audio_starts = (timestamps(frame_audio_starts)- timestamps(1)) * 1e-9;
    audio_samples_to_padd = round(duration_before_audio_starts * 44100) - audio_ind(1);
    start_padd = zeros(height(audio_data),audio_samples_to_padd);
end
if isempty(audio_end_ind)
    audio_end_ind = length(audio_data);
    frame_audio_ends = find(video_serial == audio_end_frame_id);
    duration_before_audio_end = (timestamps(end)- timestamps(frame_audio_ends)) * 1e-9;
    audio_samples_to_padd = round(duration_before_audio_end * 44100)-(audio_end_ind-audio_ind(end));
    end_padd = zeros(height(audio_data),audio_samples_to_padd);
end

audio_clip = [start_padd,audio_data(:,audio_start_ind:audio_end_ind),end_padd];


end

