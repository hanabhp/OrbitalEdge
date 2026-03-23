# OrbitalEdge

Cross-layer protocol invariants for proactive handover at the LEO satellite edge

[![SEC 2026](https://img.shields.io/badge/Venue-SEC%202026-7b3fa0?style=flat-square)](#citation)
[![Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-f0b030?style=flat-square)](https://hanapasandi.github.io/orbitaledge/demo/)
[![Tests](https://img.shields.io/github/actions/workflow/status/hanapasandi/orbitaledge/test.yml?label=tests&style=flat-square)](https://github.com/hanapasandi/orbitaledge/actions)
[![License](https://img.shields.io/badge/License-MIT-3dba90?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-555?style=flat-square)](https://python.org)

Hannah B. Pasandi, Franck Rousseau, Tamer Nadeem

UC Berkeley Sky Computing Lab / University Grenoble Alpes / Virginia Commonwealth University

Contact: hanapasandi@eecs.berkeley.edu


## What this is

LEO Direct-to-Cell satellites at 530 km cause a handover every 12 seconds for a moving
user. Every existing predictor watches RSRP and SINR. At the D2C signal floor, those
metrics are statistically indistinguishable from normal fading (Kolmogorov-Smirnov
p = 0.21, SINR near 0 dB). They provide at most 2 seconds of warning.

OrbitalEdge reads cross-layer protocol features spanning PHY through PDCP that encode
the handover signature 8 to 12 seconds in advance. The DL/UL TBS ratio drops from 0.89
to 0.63. Measurement-report density triples. RACH events co-occur with RRC
reconfigurations in a pattern that holds across four US carriers and 106 GB of real
commercial 5G diagnostic traces.

The system runs as an O-RAN Near-RT RIC rApp. It sends Conditional Handover preparation
8.3 seconds ahead, improving handover-window TCP throughput by 34.2 percent over the
3GPP Conditional Handover standard (Release 17). Inference takes 6.7 ms, within the
10 ms O-RAN control budget.


## Interactive demo

The demo runs entirely in the browser. No server, no build step, no installation.

Run locally:

```bash
git clone https://github.com/hanapasandi/orbitaledge
open orbitaledge/demo/index.html
```

Hosted version: https://hanapasandi.github.io/orbitaledge/demo/

The demo shows 5 LEO satellites orbiting at 530 km, a moving user (driving, walking,
or static), and the predictor updating in real time from protocol invariant traces. The
sidebar plots DL/UL TBS ratio, MR density, PDCCH aggregation level, and RACH
co-occurrence as the handover approaches. Use the Settings panel to configure satellite
count (3 to 9), handover interval, orbital speed, SNR stress level, and prediction
threshold.

To enable GitHub Pages on your own fork:

1. Push the repository to GitHub.
2. Go to Settings, then Pages.
3. Set source to "Deploy from branch," select main, folder "/" as the root.
4. After about 60 seconds the demo is live at your GitHub Pages URL.

The demo is a single self-contained HTML file. No build config is needed.


## Results

Evaluated on held-out Verizon carrier, NTN-emulated channel, driving mobility, 15-second
prediction horizon. All methods trained on T-Mobile, AT&T, and Sprint.

| Method | Scope | F1 | Lead time | Inference |
|---|---|---|---|---|
| OrbitalEdge (ours) | All 31 features | 0.923 | 8.3 s | 6.7 ms |
| Random Forest | All 31 features | 0.911 | 6.8 s | 3.4 ms |
| Transformer encoder | All 31 features | 0.896 | 7.6 s | 5.8 ms |
| LSTM with attention | All 31 features | 0.893 | 5.9 s | 6.2 ms |
| ProActive-HO (Lee et al., TWC 2024) | MR and PHY features | 0.877 | 4.8 s | 3.1 ms |
| SaTCP-inspired (Cao et al., INFOCOM 2023) | L2 to L4 features | 0.841 | 5.2 s | 2.4 ms |
| 3GPP Conditional Handover, Release 17 | Ephemeris only | 0.819 | 2.0 s | 0.1 ms |

Throughput improvement over 3GPP-CHO: +34.2 percent during the 5-second handover window.


## Why signal metrics fail and protocol invariants work

At the D2C signal floor, pre-handover RSRP (mean -124 dBm) overlaps with normal RSRP
(mean -121 dBm) across a +/-6 dB spread. A KS test cannot separate the two distributions
(p = 0.21). The 3GPP-CHO T1 timer therefore fires only when the satellite has already
passed the coverage boundary, leaving 2 seconds of preparation time.

The MAC scheduler begins reducing DL transport block allocations to the departing UE
8 to 12 seconds before the RRC reconfiguration fires. This is visible in every
3GPP-compliant gNB without hardware modification. The DL/UL TBS ratio drops from 0.89
to 0.63. A KS test on this feature separates the two distributions with statistic 0.41
and p less than 0.001. The 16 protocol invariants defined in the paper encode this
and related causal signatures across PHY, MAC, RLC, PDCP, and RRC.


## The 16 protocol invariants

| Group | Features | Key signature |
|---|---|---|
| A: DL/UL asymmetry | DL/UL TBS ratio, HARQ-ACK ratio, RLC PDU ratio, PDCP/RLC throughput ratio | Ratio drops 0.89 to 0.63 starting 12 seconds before the event |
| B: Gradient correlations | DL TBS gradient, UL TBS gradient, MCS gradient, RLC retransmission gradient | DL gradient turns negative at 8 seconds before the event |
| C: Causal consistency | RACH/RRC co-occurrence density, MR density, RRC reconfiguration rate, DRX interruption ratio | MR density triples at 5 seconds before the event (76 percent of total GBM importance) |
| D: PHY state | PDCCH aggregation level, RSRP proxy drop, UL power trend, CQI drop indicator | Aggregation level rises from 2.5 to 6.0 |

All 16 invariants together account for 99.9 percent of GBM feature importance across
four US carriers. Raw traffic counters contribute 0.1 percent.

The invariants are carrier-invariant (mean absolute deviation across the four carriers:
0.012) and NTN-transferable (feature shift under Doppler +/- 24 ppm and 4.76 ms
propagation delay: 0.066, well below the inter-class separation of 0.24).


## System pipeline

```
Layer 1    UE (smartphone)   gNB (macro cell)   TLE ephemeris   USRP B210
                |                   |                 |               |
                +-------------------+-----------------+---------------+
                                          |
Layer 2                         Feature extractor
                                31-dim (PHY to PDCP cross-layer KPMs)
                                16 carrier-invariant invariants + 15 raw counters
                                          |
                           +--------------+--------------+
                           |                             |
Layer 3              GBM predictor               16 protocol invariants
                     200 trees, depth 5          (Groups A, B, C, D)
                           |                             |
                           +-------> LLM reranker <------+
Layer 4                           Llama-3.2-1B, LoRA 8-bit
                                  Reasons over 20 s protocol history
                                            | 6.7 ms total
                                            v
Output              p_HO, t_lead  ->  Near-RT RIC A1 policy  ->  CHO preparation
                                      (10 ms budget)              (gNB pre-config)
```


## Repository layout

```
orbitaledge/
    src/
        feature_extractor.py       31-dim protocol feature extraction
        ntn_emulator.py            USRP B210 and GNU Radio NTN channel emulator
        predictor.py               OrbitalEdge predictor and all baselines
        run_experiments.py         Reproduces main paper numbers
        run_new_experiments.py     Six extended experiments (E1 to E6)
        run_strong_baselines.py    Nine-method baseline comparison
    demo/
        index.html                 Self-contained interactive browser demo
    tests/
        test_all.py                33 automated tests
    results/                       Written by run_experiments.py, not committed
    docs/
        hardware_setup.md          USRP B210 hardware reproduction guide
    .github/
        workflows/
            test.yml               CI: runs all tests on every push
    CITATION.cff                   Machine-readable citation
    LICENSE                        MIT
    requirements.txt
    README.md
```


## Quick start

```bash
git clone https://github.com/hanapasandi/orbitaledge
cd orbitaledge

pip install -r requirements.txt

python -m pytest tests/test_all.py -v
# Expected: 33 passed

cd src

python run_experiments.py
# Writes results/locked_results.json with all headline numbers

python run_new_experiments.py
# Writes results/locked_results_extended.json
# Covers ablation, USRP validation, scalability, sensitivity, feature importance, CDF

python run_strong_baselines.py
# Writes results/locked_baselines.json
# Nine-method comparison including recent literature
```


## Hardware reproduction

The NTN channel emulator supports software-only mode (default) and hardware mode using
two USRP B210 units. Software mode applies calibrated Doppler, delay, and path loss to
protocol feature vectors. Hardware mode activates the UHD path.

```python
from src.ntn_emulator import NTNChannelEmulator

# Software mode, no hardware required
emu = NTNChannelEmulator(orbit_alt_km=530, mode="software")

# Hardware mode, requires two USRP B210 units via USB 3.0
emu = NTNChannelEmulator(orbit_alt_km=530, mode="usrp")
```

Full setup instructions, calibration procedure, and srsRAN configuration are in
docs/hardware_setup.md.

The emulator is calibrated to Garcia-Cabeza et al., "A First Measurement Study of
T-Mobile/Starlink Direct-to-Cell Service," IEEE Communications Magazine, 2025.


## Dataset

The 106 GB cross-layer trace dataset was collected using MobileInsight-format Qualcomm
diagnostic logging on commercial UE devices across four US carriers and three mobility
conditions (static at 0 mph, walking at roughly 3 mph, and driving at 30 to 60 mph).

Raw traces are not publicly redistributed due to carrier data-use agreements. The
synthetic trace generator in feature_extractor.py reproduces the statistical properties
of the real dataset, calibrated to the characterization in the paper.

Pre-trained LLM weights (Llama-3.2-1B LoRA checkpoint, approximately 1.2 GB) are
available on request.


## Citation

If you use OrbitalEdge or its protocol invariants in your work, please cite:

```bibtex
@inproceedings{pasandi2026orbitaledge,
  author    = {Hannah B. Pasandi and Franck Rousseau and Tamer Nadeem},
  title     = {Cross-Layer Protocol Invariants for Proactive Handover
               at the {LEO} Satellite Edge},
  booktitle = {Proceedings of the 11th ACM/IEEE Symposium on Edge
               Computing (SEC)},
  year      = {2026},
  month     = oct,
}
```


## License

MIT License. See LICENSE for details.

The pre-trained LLM weights are subject to the Meta Llama 3 Community License.


## Contact

Hannah B. Pasandi
hanapasandi@eecs.berkeley.edu
Sky Computing Lab, UC Berkeley
