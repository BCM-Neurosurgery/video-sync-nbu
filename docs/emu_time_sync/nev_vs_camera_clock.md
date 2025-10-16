## Question

With a merged dataframe, do the clocks behave the same from NEV and from cameras? 


## Frequency of serial data in NEV vs. Camera JSON

### Question

Are there any differences between the sampling frequency of serials in NEV vs. in camera json?

### Methodology

Sampling frequency of serials in NEV
    
    - Random sample k NEV files from patient data that has serials
    - For each NEV, get serial dataframe, and calculate FPS of serials
    - Average FPS across k NEV files
    - Log report

Sampling frequency of serials in camera JSONs

    - Random sample k camera json files from patient data
    - For each json, calculate FPS from real_times
    - Average FPS across k camera jsons
    - Log report

### Scripts

- [Sample NEV serial FPS](../assets/nev_vs_camera_clock/sample_serial_fps.py)
- [Sample camera JSON FPS](../assets/nev_vs_camera_clock/sample_camera_json_fps.py)

### Results

NEV serial FPS
```
Per-file summary (k=20):
[INFO]   EMU-0055_subj-YFM_task-ABCD_run-01_blk-01_NSP-1.nev          -> fps_overall=29.95840700 fps_median=29.96973057 dt_mean_ms=33.380
[INFO]   EMU-0083_convo_NSP-1.nev                                     -> fps_overall=29.96042561 fps_median=29.96973057 dt_mean_ms=33.377
[INFO]   EMU-0068_subj-YFH_task-GoNoGoComplex-jumble-run-01_NSP-1.nev -> fps_overall=29.95876643 fps_median=29.96973057 dt_mean_ms=33.379
[INFO]   EMU-0019_subj-YFF_task-2-Back_Ac_run-01_NSP-1.nev            -> fps_overall=29.95907354 fps_median=29.96973057 dt_mean_ms=33.379
[INFO]   EMU-0022_subj-YFB_task-PEP_lead-RT2bHb_elec-4_NSP-1.nev      -> fps_overall=29.95994112 fps_median=29.96973057 dt_mean_ms=33.378
[INFO]   EMU-0120_subj-YFM_task-Pacman_time-20250317_175151_NSP-1.nev -> fps_overall=29.96065988 fps_median=29.96973057 dt_mean_ms=33.377
[INFO]   EMU-0004_subj-YFB_task-PEP_lead-RT1CM_elec-4_NSP-1.nev       -> fps_overall=29.96004246 fps_median=29.96973057 dt_mean_ms=33.378
[INFO]   EMU-0048_subj-YFD_task-EyesClosed_date-20240803_time-160423_NSP-1.nev -> fps_overall=29.95870531 fps_median=29.96973057 dt_mean_ms=33.379
[INFO]   EMU-0045_SAVE_NAME_NSP-1.nev                                 -> fps_overall=29.96089412 fps_median=29.96973057 dt_mean_ms=33.377
[INFO]   EMU-0038_subj-YFK_task-GoNoGoComplex-jumble-run-02_NSP-1.nev -> fps_overall=29.96072207 fps_median=29.96973057 dt_mean_ms=33.377
[INFO]   EMU-0049_SAVE_NAME_NSP-1.nev                                 -> fps_overall=29.96098052 fps_median=29.96973057 dt_mean_ms=33.377
[INFO]   EMU-0007_subj-YFD_task-4MAB_run-01_NSP-1.nev                 -> fps_overall=29.95988899 fps_median=29.96973057 dt_mean_ms=33.378
[INFO]   EMU-0025_convo_NSP-1.nev                                     -> fps_overall=29.96065397 fps_median=29.96973057 dt_mean_ms=33.377
[INFO]   EMU-0026_subj-YFN_task-EyesOpen_date-20250405_time-112613_NSP-1.nev -> fps_overall=29.95920908 fps_median=29.96973057 dt_mean_ms=33.379
[INFO]   EMU-0044_convo_NSP-1.nev                                     -> fps_overall=29.95929569 fps_median=29.96973057 dt_mean_ms=33.379
[INFO]   EMU-0032_MEM-SemOther_NSP-1.nev                              -> fps_overall=29.96079917 fps_median=29.96973057 dt_mean_ms=33.377
[INFO]   EMU-0008_subj-YFB_task-PEP_lead-LT2bHbE_elec-1_NSP-1.nev     -> fps_overall=29.96024863 fps_median=29.96973057 dt_mean_ms=33.378
[INFO]   EMU-0023_subj-YFH_task-PEP_lead-LT2aA_elec-13_NSP-1.nev      -> fps_overall=29.95849363 fps_median=29.96973057 dt_mean_ms=33.380
[INFO]   EMU-0111_subj-YFI_task-EyesOpen_date-20241020_time-161617_NSP-1.nev -> fps_overall=29.95904762 fps_median=29.96973057 dt_mean_ms=33.379
[INFO]   EMU-0023_subj-YFF_task-noisyAV_run-03_NSP-1.nev              -> fps_overall=29.95876598 fps_median=29.96973057 dt_mean_ms=33.379
[INFO] 
Aggregate statistics across sampled NEVs:
[INFO]   mean_fps_overall   = 29.95975104
[INFO]   median_fps_overall = 29.95991505
[INFO]   mean_fps_median    = 29.96973057
[INFO]   mean_dt_mean_ms    = 33.378
[INFO]   max_fps_overall     = 29.96098052 (EMU-0049_SAVE_NAME_NSP-1.nev)
[INFO]   min_fps_overall     = 29.95840700 (EMU-0055_subj-YFM_task-ABCD_run-01_blk-01_NSP-1.nev)
```

JSON Serial FPS
```
Per-file summary (k=50):
[INFO]   YFGDatafile_20240926_102339.json                             -> fps_overall=29.95883760 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFQDatafile_20250614_093817.json                             -> fps_overall=29.95918666 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFGDatafile_20240928_060717.json                             -> fps_overall=29.95838881 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFF_20240825_215540.json                                     -> fps_overall=29.95794004 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFHDatafile_20241008_145507.json                             -> fps_overall=29.95813950 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFF_20240823_200129.json                                     -> fps_overall=29.95803977 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFF_20240829_190942.json                                     -> fps_overall=29.95813950 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFQDatafile_20250617_143915.json                             -> fps_overall=29.95679325 fps_median=30.30303030 dt_mean_ms=33.381
[INFO]   YFC_20240725_202804.json                                     -> fps_overall=29.96018403 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFGDatafile_20240928_034706.json                             -> fps_overall=29.95838881 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFODatafile_20250426_223712.json                             -> fps_overall=29.95998455 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFLDatafile_20250227_200037.json                             -> fps_overall=29.95908693 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFLDatafile_20250228_193231.json                             -> fps_overall=29.95963547 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFPDatafile_20250512_181143.json                             -> fps_overall=29.96123134 fps_median=30.30303030 dt_mean_ms=33.376
[INFO]   YFKDatafile_20250216_221957.json                             -> fps_overall=29.96013416 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFB_20240506_192344.json                                     -> fps_overall=29.95888746 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFHDatafile_20241004_113351.json                             -> fps_overall=29.95823922 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFODatafile_20250424_151244.json                             -> fps_overall=29.95878773 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFSDatafile_20250718_112628.json                             -> fps_overall=29.95704254 fps_median=30.30303030 dt_mean_ms=33.381
[INFO]   YFGDatafile_20240925_071604.json                             -> fps_overall=29.95843868 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFNDatafile_20250401_171348.json                             -> fps_overall=29.95853841 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFD_20240803_163310.json                                     -> fps_overall=29.95828909 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFC_20240720_172804.json                                     -> fps_overall=29.95938613 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFODatafile_20250424_024143.json                             -> fps_overall=29.95973521 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFB_20240509_114401.json                                     -> fps_overall=29.95893733 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFJDatafile_20241109_233226.json                             -> fps_overall=29.95913680 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFQDatafile_20250611_080220.json                             -> fps_overall=29.95953573 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFPDatafile_20250506_222831.json                             -> fps_overall=29.96053313 fps_median=30.30303030 dt_mean_ms=33.377
[INFO]   YFLDatafile_20250305_132101.json                             -> fps_overall=29.95993468 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFGDatafile_20240925_115627.json                             -> fps_overall=29.95893733 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFSDatafile_20250716_202306.json                             -> fps_overall=29.95659381 fps_median=30.30303030 dt_mean_ms=33.382
[INFO]   YFE_20240814_020710.json                                     -> fps_overall=29.95888746 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFE_20240814_203842.json                                     -> fps_overall=29.95794004 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   calibration_20240918_141049.json                             -> fps_overall=29.96571646 fps_median=30.30303030 dt_mean_ms=33.371
[INFO]   YFTDatafile_20250731_084337.json                             -> fps_overall=29.95828909 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFTDatafile_20250802_173819.json                             -> fps_overall=29.95878773 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFIDatafile_20241018_194628.json                             -> fps_overall=29.95913680 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFF_20240824_232347.json                                     -> fps_overall=29.95848854 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFQDatafile_20250611_140249.json                             -> fps_overall=29.95983495 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFPDatafile_20250515_135713.json                             -> fps_overall=29.95838881 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFMDatafile_20250312_180329.json                             -> fps_overall=29.95863814 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFPDatafile_20250512_091842.json                             -> fps_overall=29.96153059 fps_median=30.30303030 dt_mean_ms=33.376
[INFO]   YFHDatafile_20241010_020801.json                             -> fps_overall=29.95848854 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFIDatafile_20241021_042105.json                             -> fps_overall=29.95943600 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFC_20240721_000837.json                                     -> fps_overall=29.95898720 fps_median=30.30303030 dt_mean_ms=33.379
[INFO]   YFKDatafile_20250215_194749.json                             -> fps_overall=29.95953573 fps_median=30.30303030 dt_mean_ms=33.378
[INFO]   YFPDatafile_20250515_214752.json                             -> fps_overall=29.95828909 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFPDatafile_20250509_031240.json                             -> fps_overall=29.96113159 fps_median=30.30303030 dt_mean_ms=33.377
[INFO]   YFPDatafile_20250514_223557.json                             -> fps_overall=29.95828909 fps_median=30.30303030 dt_mean_ms=33.380
[INFO]   YFHDatafile_20241010_140901.json                             -> fps_overall=29.95813950 fps_median=30.30303030 dt_mean_ms=33.380
[INFO] 
Aggregate statistics across sampled JSONs:
[INFO]   mean_fps_overall   = 29.95905998
[INFO]   median_fps_overall = 29.95886253
[INFO]   mean_fps_median    = 30.30303030
[INFO]   mean_dt_mean_ms    = 33.379
[INFO]   max_fps_overall     = 29.96571646 (calibration_20240918_141049.json)
[INFO]   min_fps_overall     = 29.95659381 (YFSDatafile_20250716_202306.json)
```
