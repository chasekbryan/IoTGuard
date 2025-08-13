#!/usr/bin/env python3
"""
IoTGuard: A lightweight anomaly detection tool for home IoT networks.

This script provides a simple, hands‑on example of how to build and use an
unsupervised anomaly detector for network connections. Many consumer IoT
devices expose services or make outgoing connections with little visibility
for the homeowner. Traditional host‑based tools like antivirus software
often do not run on embedded devices, and full‑fledged network intrusion
detection systems can be complex to deploy. IoTGuard fills a gap by
allowing a user to quickly build a baseline model of normal connection
patterns and then monitor for unusual behaviour using an Isolation Forest.

How it works
============
• During a training phase, IoTGuard periodically samples active network
  connections on the host machine using `psutil.net_connections()`. For each
  sampling interval it computes simple features derived from the remote and
  local addresses (the first three octets of the remote IP address, the
  remote port and the local port). These features are averaged across all
  connections to produce a single vector representing that interval. A
  collection of these vectors forms the baseline dataset.
• After training, IoTGuard fits an Isolation Forest on the baseline dataset.
  Isolation Forest is an unsupervised anomaly detection algorithm well
  suited to situations where only examples of normal behaviour are
  available. It learns to isolate points that look unusual compared to the
  baseline.
• In monitoring mode, the tool repeatedly samples current connections,
  computes the feature vector for the interval and feeds it into the
  trained model. If the model labels the new sample as an outlier, IoTGuard
  prints an alert along with the anomaly score. Otherwise it reports that
  traffic appears normal.

Limitations
-----------
IoTGuard is intended as a teaching and prototyping tool. It does not
capture payload data, cannot attribute anomalies to specific devices on
complex networks, and should only be run on networks you own or have
permission to monitor. Because the feature set is extremely simple, it may
generate false positives or miss subtle attacks. Nevertheless it
illustrates how one might build a lightweight behavioural model without
expensive hardware or proprietary software.

Usage
-----

1. Run `python iotguard.py --train 120` to collect two minutes of baseline data.
   You can adjust the duration as needed; longer baselines usually produce
   more robust models. While training, try to ensure that network traffic
   reflects typical benign usage (e.g. normal browsing, streaming, etc.).
2. Once training completes, IoTGuard automatically saves the trained model
   to the file specified by `--model` (default: `iotguard_model.joblib`).
3. To start monitoring, run `python iotguard.py --monitor`. The tool will
   load the saved model and print a message every sampling interval
   indicating whether traffic looks normal or anomalous. Use Ctrl+C to
   terminate monitoring.

Example:

```
python iotguard.py --train 60 --interval 5 --model mymodel.joblib
python iotguard.py --monitor --interval 5 --model mymodel.joblib
```

This will train for 60 seconds with 5 second sampling intervals, save the
model to `mymodel.joblib`, and then monitor every 5 seconds using the same
model.

"""

import argparse
import time
from typing import List, Optional

import numpy as np
import psutil
from sklearn.ensemble import IsolationForest
from joblib import dump, load


def extract_features() -> Optional[np.ndarray]:
    """Capture current network connections and compute a feature vector.

    The feature vector consists of three numeric values derived from
    active connections:
    1. An integer representation of the first three octets of the remote IP.
       Using only the first three octets groups connections by /24 networks
       rather than individual hosts, reducing dimensionality and partially
       anonymising addresses.
    2. The remote port number.
    3. The local port number.

    For all active connections with a remote address, IoTGuard computes the
    mean of these three values. If no connections are present, None is
    returned.

    Returns
    -------
    Optional[np.ndarray]
        A one‑dimensional NumPy array of shape (3,) containing the averaged
        features, or None if no suitable connections were found.
    """
    conns = psutil.net_connections(kind="inet")
    records: List[List[int]] = []
    for conn in conns:
        raddr = conn.raddr
        laddr = conn.laddr
        # We only consider connections with a remote address (i.e. excluding
        # listening sockets with no remote peer)
        if not raddr or not laddr:
            continue
        try:
            remote_ip = raddr.ip
            remote_port = raddr.port
            local_port = laddr.port
            octets = remote_ip.split(".")
            # For IPv4 addresses ensure there are at least 3 octets
            if len(octets) < 3:
                continue
            ip_num = int(octets[0]) * 256 * 256 + int(octets[1]) * 256 + int(octets[2])
            records.append([ip_num, remote_port, local_port])
        except Exception:
            # Skip connections with unexpected address formats
            continue
    if not records:
        return None
    arr = np.array(records, dtype=float)
    return arr.mean(axis=0)


def train_model(duration: int, interval: int, contamination: float, model_path: str) -> None:
    """Collect baseline data and train an Isolation Forest model.

    Parameters
    ----------
    duration : int
        Duration of the training period in seconds.
    interval : int
        Sampling interval in seconds.
    contamination : float
        Estimated fraction of outliers in the training data. A small value
        (e.g. 0.05) assumes that the majority of observed traffic is benign.
    model_path : str
        Path to save the trained model.
    """
    samples: List[np.ndarray] = []
    print(f"[IoTGuard] Starting training for {duration} seconds (interval {interval}s)...")
    start_time = time.time()
    while (time.time() - start_time) < duration:
        feat = extract_features()
        if feat is not None:
            samples.append(feat)
        else:
            print("[IoTGuard] Warning: no active connections observed during this interval.")
        time.sleep(interval)
    if not samples:
        raise RuntimeError("No baseline samples collected. Unable to train model.")
    X = np.stack(samples)
    # Train Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X)
    dump(clf, model_path)
    print(f"[IoTGuard] Training complete. Model saved to {model_path}.")


def monitor(interval: int, model_path: str, threshold: float) -> None:
    """Load a trained model and monitor network connections for anomalies.

    Parameters
    ----------
    interval : int
        Sampling interval in seconds.
    model_path : str
        Path to the trained model.
    threshold : float
        Anomaly score threshold. Lower values make the detector more sensitive.
        The IsolationForest decision_function returns higher scores for normal
        points and lower (negative) scores for anomalies. If the score is
        below the threshold, the sample is considered anomalous.
    """
    clf: IsolationForest = load(model_path)
    print(f"[IoTGuard] Monitoring started using model {model_path}. Press Ctrl+C to stop.")
    try:
        while True:
            feat = extract_features()
            if feat is not None:
                feat = feat.reshape(1, -1)
                score = float(clf.decision_function(feat)[0])
                if score < threshold:
                    print(f"[IoTGuard][ALERT] Anomaly detected! score={score:.4f}")
                else:
                    print(f"[IoTGuard] Normal traffic. score={score:.4f}")
            else:
                print("[IoTGuard] No active connections observed.")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[IoTGuard] Monitoring stopped by user.")


def main() -> None:
    parser = argparse.ArgumentParser(description="IoTGuard: Lightweight anomaly detection for home networks.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        metavar="SECONDS",
        type=int,
        help="Collect baseline data for the specified number of seconds and train a model."
    )
    mode_group.add_argument(
        "--monitor",
        action="store_true",
        help="Load a saved model and start monitoring for anomalies."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Sampling interval in seconds (default: 10)."
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help=(
            "Estimated fraction of outliers in the training data for IsolationForest "
            "(default: 0.05). Only used when training."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "Decision function threshold for anomaly alerts (default: 0). "
            "Lower values increase sensitivity. Only used when monitoring."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="iotguard_model.joblib",
        help="Path to save or load the model (default: iotguard_model.joblib)."
    )
    args = parser.parse_args()

    if args.train is not None:
        train_model(duration=args.train, interval=args.interval, contamination=args.contamination, model_path=args.model)
    elif args.monitor:
        monitor(interval=args.interval, model_path=args.model, threshold=args.threshold)


if __name__ == "__main__":
    main()