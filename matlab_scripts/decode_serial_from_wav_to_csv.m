function out_csv = decode_serial_from_wav_to_csv(wav_path, out_csv)
% DECODE_SERIAL_FROM_WAV_TO_CSV
% Standalone wrapper of the serial-decoding code.
% - Keeps constants, flips, taps, and byte assembly intact.
% - Works with mono/stereo/≥3 channels (auto-picks channel 3 if present).
% - Prints progress (% done, elapsed, ETA) and a final summary.
% - Writes a CSV with a SINGLE column: serial
%
% Usage:
%   decode_serial_from_wav_to_csv('/path/to/file.wav');
%   decode_serial_from_wav_to_csv('/path/to/file.wav','/path/to/out.csv');

    if nargin < 1 || isempty(wav_path)
        error('wav_path is required.');
    end
    if nargin < 2 || isempty(out_csv)
        [p, n] = fileparts(wav_path);
        if isempty(p), p = pwd; end
        out_csv = fullfile(p, [n '-serial.csv']);
    end

    % -------- Load audio (auto-select channel: prefer 3 if available) ---
    [y, fs] = audioread(wav_path); %#ok<ASGLU>  % fs unused for CSV; kept for parity
    nCh = size(y,2);
    if nCh == 0
        error('No audio channels found.');
    end
    ch = min(3, nCh);             % if mono->1, stereo->2, else->3
    ych = y(:, ch).';             % row vector

    % -------- Normalize -> threshold to binary ---------
    mn = min(ych); mx = max(ych);
    if mx > mn
        ych = (ych - mn) ./ (mx - mn);
    else
        warning('Constant channel detected; normalization degenerated.');
        ych = zeros(size(ych));
    end
    binary_signal = ych > 0.5;

    % -------- global flip ------------------------------------------
    binary_signal = flip(binary_signal);

    % -------- Constants --------------------------------
    W  = 231;                 % window length (samples)
    H  = 1100;                % stride / hop (samples)
    transition_points = 6:47:230;                % 5 bytes start positions
    offset_from_transition = [4,9,14,19,23,28,33,37]; % 8 taps per byte

    % -------- Preallocation -----------------------------------
    clipped_signal      = zeros(128, W);         % each row: one flipped window
    clipped_signal_byte = [];                    % populated in loop

    % -------- Progress setup --------------------------------------------
    N = length(binary_signal);
    est_blocks = max(1, floor((N - W) / H) + 1);
    t0 = tic; last_pct = -1; print_every_pct = 1;  % print every 1%

    % -------- Scan & decode --------------------------------
    current_ind = 1;
    count = 1;

    while current_ind < N
        current_value = binary_signal(current_ind);
        if current_value == 1
            current_ind = current_ind + 1;  % skip highs
            continue;
        else
            if current_ind + W > N
                break;
            end

            % flip each window
            clipped_signal(count,:) = flip(binary_signal(current_ind : current_ind + W - 1));

            % move to next block
            current_ind = current_ind + H;

            % extract 5 bytes x 8 bits
            clipped_signal_byte(1,count,:) = clipped_signal(count, transition_points(1) + offset_from_transition);
            clipped_signal_byte(2,count,:) = clipped_signal(count, transition_points(2) + offset_from_transition);
            clipped_signal_byte(3,count,:) = clipped_signal(count, transition_points(3) + offset_from_transition);
            clipped_signal_byte(4,count,:) = clipped_signal(count, transition_points(4) + offset_from_transition);
            clipped_signal_byte(5,count,:) = clipped_signal(count, transition_points(5) + offset_from_transition);

            % increment
            count = count + 1;

            % ---- progress update (console) ----
            pct = min(100, 100 * (count-1) / est_blocks);
            if pct - last_pct >= print_every_pct
                elapsed = toc(t0);
                est_total = elapsed * est_blocks / max(count-1, 1);
                eta = max(0, est_total - elapsed);
                fprintf('Decoding: %5.1f%% (%d/%d), elapsed %s, ETA %s\r', ...
                    pct, count-1, est_blocks, fmt_dur(elapsed), fmt_dur(eta));
                last_pct = pct;
            end
        end
    end
    fprintf('\n');  % end progress line

    % Trim to actual count
    nblocks = count - 1;
    if nblocks <= 0
        warning('No blocks decoded. CSV will be empty.');
    end

    % -------- Rebuild byte strings --------------
    if ~isempty(clipped_signal_byte)
        byte_string1 = join(string(flip(squeeze(clipped_signal_byte(1,1:nblocks,1:end-1)),2)),'',2);
        byte_string2 = join(string(flip(squeeze(clipped_signal_byte(2,1:nblocks,1:end-1)),2)),'',2);
        byte_string3 = join(string(flip(squeeze(clipped_signal_byte(3,1:nblocks,1:end-1)),2)),'',2);
        byte_string4 = join(string(flip(squeeze(clipped_signal_byte(4,1:nblocks,1:end-1)),2)),'',2);
        byte_string5 = join(string(flip(squeeze(clipped_signal_byte(5,1:nblocks,1:end-1)),2)),'',2);

        serial_id = flip(bin2dec(strcat(byte_string5, byte_string4, byte_string3, byte_string2, byte_string1)));
    else
        serial_id = zeros(0,1);
    end

    % -------- Write CSV: ONLY 'serial' column ---------------------------
    T = table(serial_id(:), 'VariableNames', {'serial'});

    out_dir = fileparts(out_csv);
    if ~isempty(out_dir) && ~isfolder(out_dir)
        mkdir(out_dir);
    end
    writetable(T, out_csv);

    % -------- Final summary ---------------------------------------------
    elapsed = toc(t0);
    fprintf('Done. Decoded %d serials → %s in %s.\n', nblocks, out_csv, fmt_dur(elapsed));
end

% ---- helper: format seconds as HH:MM:SS --------------------------------
function s = fmt_dur(t)
t = max(0, t);
hh = floor(t/3600); t = t - 3600*hh;
mm = floor(t/60);   ss = t - 60*mm;
s = sprintf('%02d:%02d:%02.0f', hh, mm, ss);
end
