# Quantifying Jamail Clinic Data Loss

This note documents the Jamail clinic serial-aligned audio integrity sweep. The batch output in `/mnt/labworlds/Provenza/auto/jamail_audio_loss` follows the same `quantify_audio_loss` helper flow used for the lounge and sleep reviews, but with `decoder_site=jamail`.

## 1. Executive snapshot

| Metric | Value |
|--------|-------|
| Recordings analyzed | 7 |
| Total analyzed duration | 40,691.20 s (≈11.3 h) |
| Estimated missing time | 2.25 s (≈0.0006 h) |
| Weighted loss (%) | 0.006% |
| Files with material loss (>0.5%) | 0 |

Six of seven sessions are gap-free; the single non-zero entry is a 2.25-second discontinuity.

## 2. Artifacts & reports

- Overall config + rollup: [overall report](../../assets/jamail_audio_loss/audio_loss_overall.json)
- Per-recording CSV summary: [summary report](../../assets/jamail_audio_loss/audio_loss_summary.csv)

## 3. Configuration (for reproducibility)

Key parameters from the overall JSON:

```json
{
  "fs_hz": 44100,
  "prefilter": true,
  "max_fwd_delta": 200,
  "local_window": 3,
  "decoder_site": "jamail",
  "room": "clinic",
  "patients": ["TRBD001", "TRBD002"],
  "discovered": 7,
  "analyzed": 7,
  "totals": {
    "n_audios": 7,
    "sum_analyzed_seconds": 40691.198,
    "sum_missing_seconds": 2.246,
    "weighted_loss_pct": 0.006
  }
}
```

Weighted loss (%) is `100 * sum_missing / sum_analyzed`.

## 4. Per-recording breakdown

| Patient | Date       | Duration (s) | Duration (h) | Missing (s) | Loss % |
|---------|------------|-------------:|-------------:|------------:|-------:|
| TRBD001 | 2025-04-15 | 6,275.96 | 1.74 | 0.000 | 0.00 |
| TRBD001 | 2025-06-02 | 5,614.42 | 1.56 | 0.000 | 0.00 |
| TRBD001 | 2025-06-13 | 6,353.20 | 1.76 | 0.000 | 0.00 |
| TRBD001 | 2025-06-16 | 3,687.29 | 1.02 | 2.246 | **0.06** |
| TRBD002 | 2025-07-07 | 5,358.08 | 1.49 | 0.000 | 0.00 |
| TRBD002 | 2025-07-22 | 7,010.41 | 1.95 | 0.000 | 0.00 |
| TRBD002 | 2025-08-05 | 6,391.84 | 1.78 | 0.000 | 0.00 |

## 5. Interpretation

- Jamail clinic audio shows effectively zero structural loss: 2.25 seconds missing across 11.3 hours.
- The only discontinuity (TRBD001 on 2025-06-16) is small and isolated; all other sessions are perfect.
- Given the extremely low weighted loss, Jamail clinic audio can be treated as clean unless future runs surface new gaps.

## 6. Reproduce locally

```bash
python -m scripts.doc_helper.quantify_audio_loss \
  --root /mnt/datalake/data/TRBD-53761 \
  --out-dir /mnt/labworlds/Provenza/auto/jamail_audio_loss \
  --site jamail \
  --room clinic
```

Outputs will mirror the JSON + CSV linked above.

---
Concise takeaway: Jamail clinic sessions are essentially gap-free; only a single 2.25-second jump was observed across the entire review.
