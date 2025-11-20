# First Fix — Gap Filling

From the [previous page](serial_quality.md), we observed that although there are many discontinuities: drops (negative deltas), forward jumps (delta > 1), and adjacent duplicates (delta = 0), intuitively, we can "repair" the sequence using a "gap-filling" algorithm.

---

## Files in this page

- Raw 1-Hour Serial: [`TRBD002_08062025-03-001.csv`](../../assets/fill_gaps/TRBD002_08062025-03-001.csv)
- Raw 1-Hour Serial Analysis: [`TRBD002_08062025-03-001.txt`](../../assets/fill_gaps/TRBD002_08062025-03-001.txt)
- Gap-filled Serial: [`TRBD002_08062025-03-001-gapfilled.csv`](../../assets/fill_gaps/TRBD002_08062025-03-001-gapfilled.csv)
- Gap-filled Serial Analysis: [`TRBD002_08062025-03-001-gapfilled.txt`](../../assets/fill_gaps/TRBD002_08062025-03-001-gapfilled.txt)

---

## Motivation

- The serial counter should increment by 1 for each frame (~30 fps), so consecutive values should form a near‑monotonic sequence.
- Many defects can be corrected without deleting or reordering rows by inferring the missing values between anchors that are likely correct.

Goal: produce a cleaned serial column with fewer discontinuities, preserving row count and original sample ranges.

---

## Intuition: midpoint passes

Rather than guessing each point independently, we run a small number of midpoint passes at different window sizes. Each pass “looks across” a fixed gap size g and nudges the center point(s) toward what the sequence should be if it were monotonic. For example, with `[1,2,3,3,5,6,7]`, slide a window of length 3; if the middle value `s[i]` is not between `s[i-1]` and `s[i+1]`, set it accordingly, yielding `[1,2,3,4,5,6,7]`.

Key ideas:

- Work only on the serial values; never drop or reorder rows.
- Use multiple gap sizes (e.g., 10, 100, 1000) so local inconsistencies get corrected at the appropriate scale.
- Keep it stable and fast: each pass is O(n) and purely functional w.r.t. row layout.

By stacking a few such passes, we can straighten out small step defects while leaving large‑scale structure intact.

---

## High‑level algorithm (pseudocode)

Input/Output contract:
- Input: CSV or DataFrame with exactly 3 columns: `serial,start_sample,end_sample`.
- Output: same schema; only `serial` values may change; length/order preserved.

Parameters:
- gaps: list of window sizes. If not provided, choose powers of 10 strictly less than the number of rows: [10, 100, 1000, …]. For n ≤ 10, no passes are applied.

```text
function gap_fill(series, gaps=None):
	s = to_int_list(series)
	if len(s) < 3:
		return s

	G = sanitize_or_autoselect_gaps(gaps, n=len(s))   # e.g., [10, 100, 1000]
	for g in G:
		# A single midpoint pass at window size g
		# (implementation detail lives in SerialFixer.apply_gap_passes_fast)
		s = midpoint_pass(s, g)

	return s

function midpoint_pass(s, g):
	# Conceptually: encourage monotone +1 progression across spans of size g.
	# One simple mental model:
	#   - compare s[i] and s[i+g]
	#   - estimate what the midpoint(s) should be if the run from i..i+g were smooth
	#   - adjust interior values slightly toward that estimate
	# Actual implementation is optimized for speed and integer stability.
	return adjusted_copy_of(s)
```

---

## Results

Analyzing with `scripts.fix.audiogapfiller` on the gap‑filled CSV, the first few rows look like:

```csv
serial,start_sample,end_sample
32896102,226,457
32896103,1735,1966
32896104,3245,3476
32896105,4755,4986
32896106,6264,6495
32896107,7773,8004
32896108,9283,9514
32896109,10792,11023
32896110,12302,12533
```

Compared to before:
```csv
serial,start_sample,end_sample
32896102,226,457
32896096,1735,1966
32896104,3245,3476
32896104,4755,4986
32896106,6264,6495
32896104,7773,8004
32896108,9283,9514
32896108,10792,11023
32896110,12302,12533
```

The analysis shows a significantly improved sequence:

```text
Values=105223  Steps=105222  ok=104404 (99.22%)
Counts: drop=320, duplicate=79, forward_jump=419, ok=104404
Adjacent duplicate events: 79
```

Compared to before:
```text
Values=105223  Steps=105222  ok=0 (0.00%)
Counts: drop=31613, duplicate=24671, forward_jump=48938
Adjacent duplicate events: 24671
```

It’s a significant improvement!!

---

## Limitations

Gap filling does not remove all drops, forward-jumps, and duplicates. An example of a remaining drop can be

```csv
...
32953058,76591478,76591709
32953059,76592987,76593218
32953060,76594496,76594727
32953348,76596471,76596702
32953120,76597982,76598213
32953122,76599491,76599722
32953123,76601001,76601232
...
```

Serial gets bumped from 32953060 to 32953348 before it drops back to 32953120. 

Another example of a duplicate and a large forward-jump after gap-filling can be

```csv
...
32897351,1708541,1708772
32897352,1710051,1710282
32897352,1711561,1711792
17179869184,1712867,1713098
34359738367,1713967,1714198
32897414,1715223,1715454
32897415,1716733,1716964
32897416,1718242,1718473
...
```

These are cases that can’t be corrected by gap filling alone because they exceed the local window constraints.

## Notes

- This pass does not remove rows or reorder time; it only adjusts `serial` values.
- It’s designed to fix local inconsistencies efficiently; pathological corruption may require additional passes (e.g., duplicate removal, drop correction) or more advanced reconstruction.
