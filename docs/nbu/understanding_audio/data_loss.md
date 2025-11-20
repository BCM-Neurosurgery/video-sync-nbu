# Data loss

After applying heuristic gap filling to the decoded serials (without changing the timeline), a natural question is: is there data loss, and if so, how much? This document walks through a concrete analysis.

## Example inputs

All examples derive from `TRBD002_08062025-03.mp3`. We use segment `TRBD002_20250806_104707` for camera `23512909` to mimic a real sync workflow. Files (stored on Elias):

- Camera JSON (chunk serials + frame IDs):
    - [camera JSON](../../assets/data_loss/TRBD002_20250806_104707.23512909.json)

- Sequence analyses of that JSON:
    - [fixed chunk serials report](../../assets/data_loss/TRBD002_20250806_104707.23512909.fixed_chunk_serials.txt)
    - [fixed frame IDs report](../../assets/data_loss/TRBD002_20250806_104707.23512909.fixed_frame_ids.txt)
- Decoded audio-side serials (gap-filled) clipped to the same 10‑min window:
    - [gap-filled CSV](../../assets/data_loss/raw-gapfilled.32948939-32966938.csv)
    - [CSV monotonicity report](../../assets/data_loss/raw-gapfilled.32948939-32966938.txt)
- Matched anchors (audio serial ↔ video serial/frame):
    - [anchors JSON](../../assets/data_loss/TRBD002_20250806_104707.23512909.23512909.anchors.json)
    - [anchors analysis](../../assets/data_loss/TRBD002_20250806_104707.23512909.23512909.anchors.txt)

## Video frame ID and serial checks

First, do the video-side frame IDs and serials show any loss?

Frame ID analysis:

```txt
Source → JSON[fixed_frame_ids]
Quick guide
-----------
• Step = ids[i+1] - ids[i]; expected step E = 1.
• ok           : diff == E    → values increase as expected.
• duplicate    : diff == 0    → adjacent repeated value (no increase).
• forward_jump : diff >  E    → skipped/missing values;
                  Total missing IDs = sum(diff - E) over all forward jumps.
• drop         : diff <  E    → value decreased (e.g., reset/rollover).
• Counts       : number of steps in each category (there are N-1 steps for N values).

Values=18000  Steps=17999  ok=17999 (100.00%)
Counts: ok=17999
Longest OK segment: start=0  length=17999 steps
```

This shows a complete 18,000 frames, with 17,999 steps all strictly increasing (no duplicates or drops).

Video-side serial analysis:

```txt
Source → JSON[fixed_chunk_serials]
Quick guide
-----------
• Step = ids[i+1] - ids[i]; expected step E = 1.
• ok           : diff == E    → values increase as expected.
• duplicate    : diff == 0    → adjacent repeated value (no increase).
• forward_jump : diff >  E    → skipped/missing values;
                  Total missing IDs = sum(diff - E) over all forward jumps.
• drop         : diff <  E    → value decreased (e.g., reset/rollover).
• Counts       : number of steps in each category (there are N-1 steps for N values).

Values=18000  Steps=17999  ok=17999 (100.00%)
Counts: ok=17999
Longest OK segment: start=0  length=17999 steps
```

Same result: video-side serials are perfectly monotonic.

## Anchor analysis (audio ↔ video)

Now we match audio serials to video serials/frame IDs using `scripts.doc_helper.collect_anchors`, producing the [anchors JSON](../../assets/data_loss/TRBD002_20250806_104707.23512909.23512909.anchors.json), then run sequence analysis to get the [anchors report](../../assets/data_loss/TRBD002_20250806_104707.23512909.23512909.anchors.txt).

```txt
=== FRAME-ID ANALYSIS (FIXED frame_id_fixed) ===
N unique frame_id_fixed: 16003
Quick guide
-----------
• Step = ids[i+1] - ids[i]; expected step E = 1.
• ok           : diff == E    → values increase as expected.
• duplicate    : diff == 0    → adjacent repeated value (no increase).
• forward_jump : diff >  E    → skipped/missing values;
                  Total missing IDs = sum(diff - E) over all forward jumps.
• drop         : diff <  E    → value decreased (e.g., reset/rollover).
• Counts       : number of steps in each category (there are N-1 steps for N values).

Values=16003  Steps=16002  ok=15946 (99.65%)
Counts: forward_jump=56, ok=15946
Total missing IDs (from forward jumps): 1996
Forward diff histogram (diff > +1):
  2:10   4:3 
  8:4    12:1
  14:1   16:1
  18:1   20:1
  24:1   32:1
  40:1   44:1
  46:1   52:1
  54:4   58:3
  60:13  62:6
  64:2 
Longest OK segment: start=7508  length=1194 steps

Top forward jumps: (index, prev, curr, diff)
  @i=  2501  56806 -> 56870  Δ=+64
  @i= 10591  65922 -> 65986  Δ=+64
  @i=  1306  55490 -> 55552  Δ=+62
  @i=  4477  59144 -> 59206  Δ=+62
  @i= 10216  65486 -> 65548  Δ=+62
```

This reveals that only 16,003 of the 18,000 expected matches are found; the rest are “forward jumps” (missing values). Zooming in:

```
  {
    "serial": 32951744,
    "audio_sample": 74873631,
    "cam_serial": "23512909",
    "frame_id": 56806,
    "frame_id_fixed": 56806
  },
  {
    "serial": 32951808,
    "audio_sample": 74881738,
    "cam_serial": "23512909",
    "frame_id": 56870,
    "frame_id_fixed": 56870
  },
```

Here, serial `32951744` maps to frame `56806`. The next matched serial is `32951808` (not `32951745`), mapping to frame `56870` — a jump of 64. Looking at the decoded audio CSV around `32951744`:

```csv
...
32951743,74872121,74872352
32951744,74873631,74873862
18958254080,74875009,74875240
8589934591,74876109,74876340
32951692,74877209,74877440
32951694,74878719,74878950
32951680,74880228,74880459
32951808,74881738,74881969
32951809,74883247,74883478
...
```

Serial `32951744` (frame `56806`) matches, but the next match is `32951808` (frame `56870`).

- Audio time delta: `(74881738 - 74873631) / 44100 ≈ 0.184 s` (audio FS = 44.1 kHz)
- Video time delta (≈30 fps): `(56870 - 56806) / 30 ≈ 2.13 s`

The ≈1.95 s discrepancy indicates missing audio between these anchors.

## Another example

Another example from the anchors json

```
  {
    "serial": 32960860,
    "audio_sample": 87129024,
    "cam_serial": "23512909",
    "frame_id": 387,
    "frame_id_fixed": 65922
  },
  {
    "serial": 32960924,
    "audio_sample": 87137256,
    "cam_serial": "23512909",
    "frame_id": 451,
    "frame_id_fixed": 65986
  },
```

Serial `32960860` matches, but the next match is `32960924`. Inspecting the CSV:

```csv
...
32960859,87127515,87127746
32960860,87129024,87129255
32960844,87130534,87130765
32960838,87132043,87132274
32960832,87133553,87133784
32960924,87137256,87137487
32960925,87138766,87138997
...
```

After `32960860`, the sequence regresses (`32960844`, `32960838`, `32960832`) before resuming at `32960924`, `32960925`, …. The audio delta is `(87137256 - 87129024) / 44100 ≈ 0.187 s`, while the video delta is `(65986 - 65922) / 30 ≈ 2.13 s`, again suggesting ≈1.94 s of missing audio.

## Quantifying audio loss

To quantify audio loss, the idea is to find all those "unmatched" locations within the decode CSV, calculate the actual time difference within the audio, the time difference it should be if there were not serial gaps, differencing those 2 to get the estimated time loss, and accumulate them. To provide a more accurate estimate that resembles the frequency of the realistic serial stream, instead of using 1470 (44100hz / 30fps) in calculation, we will actually use the local median, which is around 1510 samples (which means in reality the audio gets serials in roughly 44100 / 1510 = 29.21 fps). The algorithm and implementation details are all within `scripts.analysis.audio_loss_analysis`, the report for this particular 10-min example is at [`audio loss report`](../../assets/data_loss/raw-gapfilled.32948939-32966938.audio_loss.json). 

An example:

```json
  "gaps": [
    {
      "index": 173,
      "prev_serial": 32949112,
      "curr_serial": 32949136,
      "prev_start_sample": 71343198,
      "prev_end_sample": 71343429,
      "curr_start_sample": 71349876,
      "curr_end_sample": 71350107,
      "diff": 24,
      "observed_ms": 151.42857142857142,
      "ideal_ms": 821.7687074829932,
      "missing_ms": 670.3401360544218,
      "local_period_samples": 1510
    },
```

That means serial jumps from 32949112 to 32949136, looking in the csv

```csv
...
32949110,71340180,71340411
32949111,71341689,71341920
32949112,71343198,71343429
32949112,71344708,71344939
32949136,71349876,71350107
32949176,71351386,71351617
32949177,71352894,71353125
32949178,71354404,71354635
...
```

The first 32949112 is matched to a video's serial, then 32949136 is matched, providing a gap of 24. The actual time in audio is `(71349876 - 71343198) / 1510 = 0.1514` seconds, but it should be `24 / (44100 / 1510) = 0.8218` seconds, leaving a `0.8218 - 0.1514 = 0.6704` seconds of audio loss.

Together, those many unmatched gaps accumulate to a 12% audio loss

```json
  "summary": {
    "values_kept": 16003,
    "steps": 16002,
    "ok_steps": 15947,
    "forward_jumps": 55,
    "ok_ratio": 0.9965629296337958,
    "total_missing_seconds": 66.23192743764173,
    "analyzed_seconds": 549.8451020408163,
    "loss_share_pct": 12.045561048341426
  },
```
