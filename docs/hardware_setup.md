# Hardware Setup Guide

This document describes how to reproduce the NTN channel emulation testbed used in the
OrbitalEdge paper using two Ettus Research USRP B210 units.


## Hardware requirements

| Component | Specification |
|---|---|
| SDR units | 2x Ettus Research USRP B210 |
| Antennas | VERT900, covers 900 MHz to 1.8 GHz including Band 25 UL at 1912.5 MHz |
| Host PC | Ubuntu 22.04 LTS, one USB 3.0 port per USRP |
| RF cable | SMA male to SMA male, 30 cm |
| Attenuator | 30 dB SMA inline attenuator, for example Mini-Circuits VAT-30+ |
| USB cables | 2x USB 3.0 Type-A to Type-B |


## Software requirements

```bash
sudo apt install uhd-host python3-uhd libuhd-dev
uhd_images_downloader

sudo apt install gnuradio gnuradio-dev python3-gnuradio

git clone https://github.com/srsRAN/srsRAN_4G
cd srsRAN_4G && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && sudo make install

pip install uhd numpy scipy
```


## Physical connections

```
USRP-A  TX port A  ->  30 dB attenuator  ->  RX port A  USRP-B
           USB 3.0                                USB 3.0
              |                                      |
     Host PC running srsRAN UE          Host PC running srsRAN gNB
              |                                      |
              +----------ethernet loopback-----------+
                                  |
                     GNU Radio NTN impairment block
                     (applied in software between TX and RX)
```


## Enabling hardware mode

```python
from src.ntn_emulator import NTNChannelEmulator

# Requires both USRPs connected via USB 3.0
emu = NTNChannelEmulator(
    orbit_alt_km=530,
    mode="usrp",
    seed=42,
    usrp_tx_addr="addr=192.168.10.2",
    usrp_rx_addr="addr=192.168.10.3",
)
```


## Calibration

The emulator is calibrated to Garcia-Cabeza et al. (IEEE CommMag 2025) T-Mobile and
Starlink D2C measurements. To verify your setup:

```bash
cd src
python ntn_emulator.py --validate
```

Expected output within tolerance:

```
RSRP median:    -121.0 dBm    (reference: -121.0, tolerance: +/- 5 dB)   PASS
Delay range:    [1.83, 7.19] ms   (reference: [1.77, 6.83] ms)           PASS
Doppler max:    25.35 ppm   (reference: 24.0 ppm, tolerance: 15 percent)  PASS
TA range:       [1800, 7200] us   (530 km orbit geometry)                 PASS
```


## Known limitation with srsRAN

srsRAN 23.11 does not implement 3GPP Release 17 condEventT1 or condEventD1, which are
the NTN-specific Conditional Handover triggers. Handovers in this testbed are injected
via RRC reconfiguration message injection at empirical 12-second intervals, as
documented in the paper (Section V, footnote 3). Full 5G NR-NTN Release 18 support is
expected in a future srsRAN release.


## Troubleshooting

USRP not detected:

```bash
uhd_find_devices
uhd_usrp_probe
```

Dropped samples or low throughput: use USB 3.0 ports, not USB 2.0, and set the CPU
governor to performance mode.

```bash
sudo cpupower frequency-set -g performance
```

If needed, reduce the sample rate by changing 7.68e6 to 3.84e6 in ntn_emulator.py.

UE cannot attach to the srsRAN gNB: verify Band 25 center frequencies in ue.conf
(UL: 1912.5 MHz, DL: 1930.1 MHz).


## Contact

For hardware reproduction questions, open an issue on GitHub or email the corresponding
author at hanapasandi@eecs.berkeley.edu.
