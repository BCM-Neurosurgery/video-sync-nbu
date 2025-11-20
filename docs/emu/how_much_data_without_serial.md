## Question

How much data collected in EMU is without serial? 

## Methodology

[Script](../assets/how_much_data_without_serial/check_nev_serials.py) here randomly samples stitched NEV files under `/mnt/stitched/EMU-18112/` for each patient, parses NEV files, and checks if the NEV files contain serials. 

## Results

```
Summary (patient, checked, with_129):
       YEY    1    1
       YEZ    1    1
       YFA    1    1
       YFB    1    1
       YFC    1    1
       YFD    1    1
       YFE    1    1
       YFF    1    1
       YFG    1    1
       YFH    1    1
       YFI    1    1
       YFJ    1    1
       YFK    1    1
       YFL    1    1
       YFM    1    1
       YFN    1    1
       YFO    1    1
       YFP   10    0
       YFQ   10    0
       YFR   10    0
       YFS   10    0
       YFT   10    0

Patients with NO InsertionReason=129 in sampled NEVs:
  - YFP
  - YFQ
  - YFR
  - YFS
  - YFT
```

Patients YFP, YFQ, YFR, YFS, YFT do not have serials in the stitched NEV files.
