"""
Input

- video
    - <SEGMENT_ID>_<TIMESTAMP>.<CAM_SERIAL1>.mp4
    - <SEGMENT_ID>_<TIMESTAMP>.json
    ...
    - <SEGMENT_ID>_<TIMESTAMP>.<CAM_SERIAL2>.mp4
    - <SEGMENT_ID>_<TIMESTAMP>.json
    ...

- YFP (patient_id)
    - EMU-0088_convo (task_id)
        - EMU-0088_convo_NSP-1.nev
        - EMU-0088_convo_NSP-1.ns5
        - ... other files

    - EMU-0100_convo (task_id)
        - EMU-0100_convo_NSP-1.nev
        - EMU-0100_convo_NSP-1.ns5
        - ... other files

Output
- out
    - YFP
        - EMU-0088_convo
            - work
            - synced_video
            - ...
"""

from scripts.models import StitchedTask


if __name__ == "__main__":
    # 1. TODO pass in an patient input and output dir outlined above in docs
    # and optionally, a list of keywords to filter tasks (convo, etc)
    # and a string of "roommic1" or "roommic2" to choose which audio channel to use
    # populate the StitchedTask dataclass lists
    # scripts that can be helpful: scripts.index, scripts.parsers

    # 2. TODO for each StitchedTask, find the range of chunk serial numbers (if exists)
    # in the DIGIEVTS of the NEV file. This will be used to filter the video segments

    # 3. TODO go through all video segments, find the serial range for each segment (if exists)
    # so that we find the segments that overlap with the NEV chunk serial range
    # scripts that can be helpful: scripts.index

    # 4. TODO now that we know which video segments to use, we need to first match the timestamps
    # in the NS5 audio channel to NEV DIGIEVTS timestamps (maybe matching timestamps, instead of UTC time is fine),
    # because they both record on BlackRock and they share the same clock.
    # Then extract and save that audio to a wav file

    # 5. TODO now go through each video segment, clip the part of the video by matching the serial from video's json
    # to the NEV DIGIEVTS if needed, then pad the frames if needed

    # 6. TODO finally, merge all the clipped and padded video segments, and add the extracted audio
    pass
