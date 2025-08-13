# IoTGuard
**A Lightweight Anomaly Detection Tool for Home IoT Networks**

## Overview
IoTGuard is an open-source Python tool that uses **unsupervised machine learning** to detect unusual network behavior from IoT devices in a home network. It learns a baseline of “normal” traffic and flags deviations that could indicate compromise or misuse.

Many consumer IoT devices lack robust security and can be exploited for botnets, unauthorized data collection, or lateral movement inside a network. Traditional host-based tools are often too heavy for IoT devices themselves, and enterprise IDS solutions are overkill for home use. IoTGuard fills this gap with a lightweight, privacy-friendly monitor that runs on a computer you control.

---

## Features
- **Unsupervised learning** with `IsolationForest` from **scikit-learn** — no attack data required.
- **Passive network monitoring** using `psutil` (safe for educational and home lab use).
- **Two modes**:
  - **Training Mode** — learns baseline network activity.
  - **Monitoring Mode** — flags activity outside baseline patterns.
- **Model persistence** with `joblib` so training is done once and reused.
- Works on **any network-connected machine** (e.g., Raspberry Pi on your router’s LAN).

---

## Installation
IoTGuard requires Python 3.8+ and the following packages:

```bash
pip install psutil scikit-learn pandas numpy joblib
```

---

## Usage

### 1. Training Phase
Run training to collect baseline network behavior:
```bash
python iotguard.py --train 120 --interval 10 --model iotguard_model.joblib
```
- `--train` — total duration in seconds to train (120 seconds in example).
- `--interval` — how often to sample connections (seconds).
- `--model` — path to save trained model.

### 2. Monitoring Phase
Run continuous monitoring against the saved model:
```bash
python iotguard.py --monitor --interval 10 --model iotguard_model.joblib --threshold -0.1
```
- `--monitor` — enables monitoring mode.
- `--interval` — sampling interval in seconds.
- `--threshold` — anomaly score cutoff; lower values increase sensitivity.

---

## Example Workflow
```bash
# Train for 5 minutes
python iotguard.py --train 300 --interval 10 --model iotguard_model.joblib

# Monitor with moderate sensitivity
python iotguard.py --monitor --interval 10 --model iotguard_model.joblib --threshold -0.1
```

---

## Notes & Recommendations
- **Where to run**: on a machine that sees all IoT traffic — e.g., a Pi connected to a mirrored router port or serving as a gateway.
- **Privacy**: IoTGuard only records metadata (IP addresses, ports, connection counts) — no payloads or content.
- **Model refresh**: retrain periodically if you add new devices or your normal traffic pattern changes.
- **Legal**: only monitor networks you own or have explicit permission to analyze.

---

## License
- License: **GNU General Public License v3.0 (GPL-3.0)**.
