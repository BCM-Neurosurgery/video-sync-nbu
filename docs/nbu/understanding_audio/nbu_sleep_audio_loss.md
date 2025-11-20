# Quantifying Sleep-Room Data Loss

This note captures the latest serial-aligned audio integrity sweep for the NBU sleep room. The batch run lives under `/mnt/labworlds/Provenza/auto/NBU_sleep_audio_loss` and follows the same pipeline used for the lounge review, but with `decoder_site=nbu_sleep`.

## 1. Executive snapshot

| Metric | Value |
|--------|-------|
| Recordings analyzed | 13 |
| Total analyzed duration | 792,739.19 s (≈220.2 h) |
| Estimated missing time | 12,842.51 s (≈3.57 h) |
| Weighted loss (%) | 1.62% |
| Files with material loss (>0.5%) | 2 |

Two outlier sessions dominate the loss figure; the remainder are effectively clean (<0.06%).

## 2. Artifacts & reports

- Overall config + rollup: [overall report](../../assets/nbu_sleep_audio_loss/audio_loss_overall.json)
- Per-recording CSV summary: [summary report](../../assets/nbu_sleep_audio_loss/audio_loss_summary.csv)

## 3. Configuration (for reproducibility)

Key parameters from the overall JSON:

```json
{
  "fs_hz": 44100,
  "prefilter": true,
  "max_fwd_delta": 200,
  "local_window": 3,
  "decoder_site": "nbu_sleep",
  "room": "sleep",
  "patients": ["TRBD001", "TRBD002"],
  "discovered": 13,
  "analyzed": 13,
  "totals": {
    "n_audios": 13,
    "sum_analyzed_seconds": 792739.185,
    "sum_missing_seconds": 12842.508,
    "weighted_loss_pct": 1.62
  }
}
```

Weighted loss (%) is still computed as `100 * sum_missing / sum_analyzed`.

## 4. Per-recording breakdown

| Patient | Date | Duration (s) | Duration (h) | Missing (s) | Loss % |
|---------|------|-------------:|-------------:|------------:|-------:|
| TRBD001 | 2025-04-16 | 5,449.90 | 1.51 | 2.59 | 0.05 |
| TRBD001 | 2025-04-16 | 77,590.15 | 21.55 | 0.03 | 0.00 |
| TRBD001 | 2025-06-05 | 27,997.11 | 7.78 | 15.98 | 0.06 |
| TRBD001 | 2025-06-11 | 88,088.67 | 24.47 | 7.51 | 0.01 |
| TRBD001 | 2025-06-17 | 59,929.05 | 16.65 | 10.17 | 0.02 |
| TRBD001 | 2025-06-18 | 10,954.37 | 3.04 | 2.88 | 0.03 |
| TRBD001 | 2025-06-18 | 17,593.96 | 4.89 | 2,247.48 | 12.77 |
| TRBD002 | 2025-05-22 | 79,340.36 | 22.04 | 10,528.66 | 13.27 |
| TRBD002 | 2025-07-08 | 174,599.84 | 48.50 | 2.18 | 0.00 |
| TRBD002 | 2025-07-24 | 66,011.01 | 18.34 | 4.30 | 0.01 |
| TRBD002 | 2025-07-24 | 10,812.59 | 3.00 | 0.07 | 0.00 |
| TRBD002 | 2025-08-07 | 87,186.08 | 24.22 | 10.34 | 0.01 |
| TRBD002 | 3035-08-07 | 87,186.08 | 24.22 | 10.34 | 0.01 |

The final line reflects a duplicate export of the 2025-08-07 session through an alternate folder name; metrics are identical, and the aggregate JSON still reports 13 analyzed audios.

## 5. Interpretation

* TRBD002 on **2025-05-22** accounts for ~82% of the missing time (≈2.92 h lost, 13.3% of the session).
* TRBD001 on **2025-06-18** (file `TRBD001_06182025-03.mp3`) contributes another ~0.62 h lost (12.8%).
* All other sessions show negligible structural loss; most fall below 0.06%.

These two sessions merit a deeper dive (gap visualizations, raw WAV spot checks, serial stability) before treating the channel as production-worthy.

## 6. Reproduce locally

```bash
python -m scripts.doc_helper.quantify_audio_loss \
  --root /mnt/datalake/data/TRBD-53761 \
  --out-dir /mnt/labworlds/Provenza/auto/NBU_sleep_audio_loss \
  --site nbu_sleep \
  --room sleep
```

Outputs will mirror the JSON + CSV linked above.

---
Concise takeaway: overall loss remains low (1.6%), but one TRBD002 sleep-room session and one TRBD001 session exhibit multi-hour gaps that drive the average and deserve focused remediation.
