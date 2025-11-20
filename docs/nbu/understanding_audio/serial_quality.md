# Serial Quality

After decoding serials from the serial audio, a natural question is: what’s the data quality? In a perfect scenario, we expect a monotonic sequence—but is that the case? To show an example representative of the general data quality, we use the **first 1‑hour slice of `TRBD002_08062025-03.mp3`** and include the decoded CSV and its text analysis report so readers can reproduce the findings.

> Files for this example are checked into the repo under: `../../assets/serial_quality/`

- CSV: [`assets/serial_quality/TRBD002_08062025-03-001.csv`](../../assets/serial_quality/TRBD002_08062025-03-001.csv)

- Text report: [`assets/serial_quality/TRBD002_08062025-03-001.txt`](../../assets/serial_quality/TRBD002_08062025-03-001.txt)


## CSV schema

| column | meaning |
|------|--------|
| `serial` | The decoded integer for this block/window of the serial track. |
| `start_sample` | Start index (audio sample) of the detected window. |
| `end_sample` | End index (audio sample) of the detected window. |
| `delta_serial` | Derived here: `serial[i] - serial[i-1]` (first row is NA). |
| `step` | Classification of `delta_serial`: `ok` (+1), `duplicate` (0), `forward_jump` (>+1), `drop` (<0). |


## Definition of discontinuities

To quickly analyze the monotonicity of the stream, we standardize the quality analysis with `scripts/analysis/serial_analysis.py` and `scripts/analysis/csv_serial_analysis.py`. Based on every pair of adjacent values, we categorize four outcomes:
```text
- ok        : diff == +1
- duplicate : diff == 0                 (adjacent equal values)
- forward   : diff > +1  (we also sum total_missing_ids = sum(diff-1))
- drop      : diff < 0
```

## Quick look at the data

A preview of the first few rows (derived `delta_serial` and `step` added for illustration):


|   row |   serial |   start_sample |   end_sample |   delta_serial | step         |
|------:|---------:|---------------:|-------------:|---------------:|:-------------|
|     0 | 32896102 |            226 |          457 |            nan | first        |
|     1 | 32896096 |           1735 |         1966 |             -6 | drop         |
|     2 | 32896104 |           3245 |         3476 |              8 | forward_jump |
|     3 | 32896104 |           4755 |         4986 |              0 | duplicate    |
|     4 | 32896106 |           6264 |         6495 |              2 | forward_jump |
|     5 | 32896104 |           7773 |         8004 |             -2 | drop         |
|     6 | 32896108 |           9283 |         9514 |              4 | forward_jump |
|     7 | 32896108 |          10792 |        11023 |              0 | duplicate    |
|     8 | 32896110 |          12302 |        12533 |              2 | forward_jump |
|     9 | 32896032 |          13811 |        14042 |            -78 | drop         |


**Observations**:

- A glance at the difference of the sample indices shows they’re roughly 1500 samples apart, consistent with 44,100 Hz audio where each serial is sent at ~30 fps from Arduino: 44,100 / 30 ≈ 1,470 samples.

- In the first 10 rows, we notice **drops** (negative deltas), **forward jumps** (delta > 1), and **adjacent duplicates** (delta = 0). A healthy stream would be nearly all `ok` (+1) steps. So what about the quality for this entire 1-hour clip? How many of those discontinuities do we have? Can we quantify them?

- We also notice that although this stream looks unusable, we might actually be able to **repair** it. For example, 32896096 should really be 32896103 based on the previous value and its next value, and the third 32896104 should really be 32896107. We know this with high confidence because the serials should be monotonic and the sample indices look correctly spaced. This suggests we can write scripts to fix the stream. More on this later.

## Quality Summary

By running `python -m scripts.analysis.csv_serial_analysis` on the CSV, we can assess the data quality for the entire 1-hour clip:

- Total values (rows): **105,223**  → Steps (N−1): **105,222**

- `ok` (+1): **0**  (0.00%)

- `forward_jump` (>+1): **48,938**  (46.51%)

- `drop` (<0): **31,613**  (30.04%)

- `duplicate` (=0): **24,671**  (23.45%)


**Interpretation**: The raw serial stream is **not monotonic**: nearly half of all steps are forward jumps, ~30% are drops, and ~23% are adjacent duplicates. The `ok` rate is **0%**, meaning there is no contiguous +1 run in this hour.

## Example: duplicates

The first few lines of adjacent discontinuities in the report are:
```text
Adjacent duplicate events: 24671
Adjacent-duplicate values (value:count):
  32896104:1  32896108:1
  32896112:1  32896116:1
  32896120:1  32896128:2
  32896132:1  32896136:1
  32896140:1  32896144:1
  ...
```

That means `32896104` is repeated once, `32896128` is repeated twice, etc. Zooming into the CSV, we see
```csv
...
32896096,1735,1966
32896104,3245,3476
32896104,4755,4986
32896106,6264,6495
...
```

```csv
...
32896000,37964,38195
32896128,39473,39704
32896128,40983,41214
32896130,42492,42723
...
```

```csv
...
32896140,228160,228391
32896128,229671,229902
32896128,231180,231411
32896256,232689,232920
...
```

## Example: forward jumps

Another example is a histogram of forward jumps in the decoded serial:
```text
Total missing IDs (from forward jumps): 2174372008059
Forward diff histogram (diff > +1):
  2:22109        3:1          
  4:9693         6:146        
  8:5277         10:101       
  12:1106        14:56        
  16:2449        18:28        
  20:727         24:64
  ...  
```

That means a difference of 2 appears 22,109 times, a difference of 10 appears 101 times, etc. Top forward jumps are:

```text
Top forward jumps: (index, prev, curr, diff)
  @i= 10864  0 -> 34359738367  Δ=+34359738367
  @i= 18145  0 -> 34359738367  Δ=+34359738367
  @i= 22382  0 -> 34359738367  Δ=+34359738367
  @i= 35740  0 -> 34359738367  Δ=+34359738367
  @i= 94195  0 -> 34359738367  Δ=+34359738367
```

This means that at index 10,864, the serial jumps from 0 to 34,359,738,367. In the CSV, we see:

```csv
...
32907776,16396958,16397189
0,16398225,16398456
34359738367,16399325,16399556
32907964,16400434,16400665
...
```

### Example: drops

Top drops are also provided in the report:
```text
Top drops: (index, prev, curr, diff)
  @i=  1136  34359738367 -> 32897414  Δ=-34326840953
  @i= 10865  34359738367 -> 32907964  Δ=-34326830403
  @i= 18146  34359738367 -> 32916294  Δ=-34326822073
  @i= 22383  34359738367 -> 32921058  Δ=-34326817309
  @i= 35741  34359738367 -> 32936110  Δ=-34326802257
```

Zooming into the CSV, we see:

```csv
...
17179869184,1712867,1713098
34359738367,1713967,1714198
32897414,1715223,1715454
32897408,1716733,1716964
...
```

## Takeaways

- The raw CSV often has poor data quality. Without a post-fixing script, it’s barely usable.
