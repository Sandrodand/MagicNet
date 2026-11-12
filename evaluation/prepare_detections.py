"""Save the detector alarms used by the campaign."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd

from detectors.detector_simulator import DetectorSimulator
from evaluation.buffer import Buffer
from evaluation.default_parameters import set_adwin_delta, set_delay, set_seq_len
from evaluation.parameter_config import Config
from evaluation.prepare_base_learner import campaign_streams, save_pickle
from evaluation.run_campaign import run_jobs


def prepare_detections(path, detector, destination, params):
    """Run one detector and save its alarms."""
    if isinstance(detector, list):
        precision, recall = detector
        simulator = DetectorSimulator(
            dataset=path,
            precision=precision,
            recall=recall,
            min_delay=params["DETECTOR_SIMULATOR_MIN_DELAY"],
            max_delay=params["DETECTOR_SIMULATOR_MAX_DELAY"],
        )
        detections = sorted(set(simulator.drifts))
    elif detector == "sentinel":
        with open(f"{path}.csv", encoding="utf-8", newline="") as file:
            total_points = sum(1 for _ in csv.DictReader(file))
        columns = pd.read_csv(f"{path}.csv", nrows=0).columns
        converters = {name: float for name in columns if name not in ("task", "target", "batch")}
        converters.update(task=int, target=int)
        dataset = Path(path).name.lower()
        seq_len = params["SEQ_LEN_PARAM"]
        if seq_len is None:
            seq_len = set_seq_len(dataset)
        delay = params["DELAY_PARAM"]
        if delay is None:
            delay = set_delay(dataset)

        cfg = Config()
        cfg.set_params(
            path=str(path),
            converters=converters,
            delta=set_adwin_delta(dataset),
            ta_order=min(seq_len - 1, params["MAX_OLD_LABELS"]),
        )
        sentinel = cfg.create_drift_detector()
        buffer = Buffer(delay)
        detections = []
        print(f"{dataset}: Sentinel started")
        for index, (x, y) in enumerate(cfg.create_iter_csv()):
            x.pop("task", None)
            x.pop("batch", None)
            point = buffer.enqueue((x, y))
            if point is not None and sentinel.update(*point):
                detections.append(index)
            processed = index + 1
            if processed % 100 == 0 or processed == total_points:
                print(f"{dataset} Sentinel {processed}/{total_points}")
    else:
        tasks = pd.read_csv(f"{path}.csv", usecols=["task"])["task"]
        detections = [i for i in range(1, len(tasks)) if tasks.iloc[i] != tasks.iloc[i - 1]]

    save_pickle(Path(destination), [detections])
    print(f"Saved {destination} ({len(detections)} detections)")


def main():
    """Queue missing detection files for all streams and detector conditions."""
    jobs = []
    seen = set()
    for params, detector, name, path in campaign_streams():
        if isinstance(detector, list):
            precision, recall = detector
            condition = f"{precision * 100:g}prec_{recall * 100:g}rec"
        elif detector == "sentinel":
            condition = "sentinel"
        elif detector is False or detector == "supervised":
            condition = "supervised"
        else:
            raise ValueError(f"Unknown detector: {detector}")

        results = Path("performance") / params["PATH_PERFORMANCE"]
        root = Path(params["PATH_LOAD_DETECTIONS"] or results)
        filename = "drifts.pkl"
        if os.path.abspath(root) == os.path.abspath(results):
            filename = "prepared_drifts.pkl"
        job_name = f"{name}_{condition}"
        destination = root / job_name / filename
        if destination.exists() or destination.resolve() in seen:
            continue
        seen.add(destination.resolve())
        jobs.append((job_name, {
            "path": str(path),
            "detector": detector,
            "destination": str(destination),
            "params": params,
        }))
    with TemporaryDirectory() as directory:
        files = []
        for name, params in jobs:
            path = Path(directory) / f"{name}.json"
            if path.exists():
                path = Path(directory) / f"{name}_{len(files)}.json"
            path.write_text(json.dumps(params), encoding="utf-8")
            files.append(path)
        return run_jobs("evaluation.prepare_detections", files, Path("outputs/prepare_detections"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.config is None:
        sys.exit(main())
    with open(args.config, encoding="utf-8") as file:
        prepare_detections(**json.load(file))
