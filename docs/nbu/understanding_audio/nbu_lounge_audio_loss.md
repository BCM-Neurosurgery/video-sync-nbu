# Quantifying NBU Lounge Data Loss

So how much data loss exactly do we have in the lounge? This note summarizes how much serial‑aligned audio timing is effectively "missing" across analyzed **NBU lounge** recordings. The helper script `scripts.doc_helper.quantify_audio_loss` batch‑processes each serial audio file found through `/mnt/datalake/data/TRBD-53761` and emits JSON + CSV summaries.

## 1. Executive snapshot

| Metric | Value |
|--------|-------|
| Recordings analyzed | 14 |
| Total analyzed duration | 376,532.8 s (≈104.0 h) |
| Estimated missing time | 7,146.35 s (≈1.98 h) |
| Weighted loss (%) | 1.90% |
| Files with material loss (>0.5%) | 3 |

Most recordings show near‑zero structural loss; the overall 1.9% is driven almost entirely by three outliers.

## 2. Artifacts & reports

- Overall config + rollup: [overall report](../../assets/how_much_data_loss/audio_loss_overall.json)
- Per‑recording CSV summary: [summary report](../../assets/how_much_data_loss/audio_loss_summary.csv)

## 3. Configuration (for reproducibility)

Key parameters embedded in the JSON (trimmed for readability):

```json
{
  "fs_hz": 44100,
  "prefilter": true,
  "max_fwd_delta": 200,
  "local_window": 3,
  "patients": ["TRBD001", "TRBD002"],
  "discovered": 14,
  "analyzed": 14,
  "totals": {
    "n_audios": 14,
    "sum_analyzed_seconds": 376532.845,
    "sum_missing_seconds": 7146.35,
    "weighted_loss_pct": 1.898
  }
}
```

Weighted loss (%) is computed as:

```python
weighted_loss = 100.0 * sum_missing / sum_analyzed
```

## 4. Per‑recording breakdown

| Patient | Date       | Duration (s) | Duration (h) | Missing (s) | Loss % |
|---------|------------|-------------:|------------:|------------:|-------:|
| TRBD001 | 2025-04-16 | 36985.01 | 10.27 | 4055.40 | **10.96** |
| TRBD001 | 2025-06-03 | 39263.53 | 10.91 | 0.14 | 0.00 |
| TRBD001 | 2025-06-04 | 48083.01 | 13.36 | 2.84 | 0.01 |
| TRBD001 | 2025-06-10 | 22586.69 | 6.27  | 0.94 | 0.00 |
| TRBD001 | 2025-06-10 | 16117.47 | 4.48  | 0.00 | 0.00 |
| TRBD001 | 2025-06-11 | 11384.07 | 3.16  | 0.24 | 0.00 |
| TRBD001 | 2025-06-17 | 41946.43 | 11.65 | 0.64 | 0.00 |
| TRBD001 | 2025-06-18 | 5230.64  | 1.45  | 0.10 | 0.00 |
| TRBD002 | 2025-05-21 | 43088.39 | 11.97 | 0.87 | 0.00 |
| TRBD002 | 2025-07-10 | 36716.97 | 10.20 | 0.72 | 0.00 |
| TRBD002 | 2025-07-10 | 6807.50  | 1.89  | 0.16 | 0.00 |
| TRBD002 | 2025-07-23 | 43543.35 | 12.10 | 3.05 | 0.01 |
| TRBD002 | 2025-08-07 | 22267.96 | 6.19  | 2685.73 | **12.06** |
| TRBD002 | 2025-08-07 | 2511.82  | 0.70  | 395.50 | **15.75** |

Observations:

* Only two calendar dates (Apr 16 and Aug 07) contribute virtually all loss.
* The smaller Aug 07 file (≈0.70 h) shows very high relative loss (15.7%).
* Other sessions are effectively clean (<0.02%).

## 5. Interpretation

The overall 1.9% masks a bimodal pattern: nearly all sessions are clean; a small minority show sustained discontinuities (serial forward jumps).

## 6. Reproduce locally

```bash
python -m scripts.doc_helper.quantify_audio_loss \
  --root /mnt/datalake/data/TRBD-53761 \
  --out-dir /mnt/labworlds/Provenza/auto/TRBD_audio_loss \
  --site nbu_lounge
```

Outputs will mirror the JSON + CSV linked above.

---
Concise takeaway: overall data integrity is strong; loss is localized to a small number of outlier recordings that warrant targeted investigation.
