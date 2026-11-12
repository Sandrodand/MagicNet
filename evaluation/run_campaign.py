"""Run a campaign using one temporary parameter file per job."""

import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time

from evaluation.test import PARAMETERS


def create_jobs(campaign, directory):
    """Write the parameter files and return their paths in launch order."""
    defaults = PARAMETERS.copy()
    for key in defaults:
        if key in campaign:
            defaults[key] = campaign[key]

    jobs = []
    for row in campaign["campaign"]:
        detector = row["detector"]
        precision = recall = None
        if isinstance(detector, list):
            precision, recall = detector
            condition = f"{precision * 100:g}prec_{recall * 100:g}rec"
            priority = (0 if detector == [1, 1] else 2, -precision, -recall)
        elif detector == "sentinel":
            condition = "sentinel"
            priority = (1, 0, 0)
        elif detector is False or detector == "supervised":
            condition = "supervised"
            priority = (3, 0, 0)
        else:
            raise ValueError(f"Unknown detector: {detector}")

        configurations = row["configurations"]
        if isinstance(configurations, dict):
            first, last = configurations["range"]
            configurations = range(first, last + 1)
        for number in configurations:
            stream = f"{row['dataset']}_{number}conf"
            name = f"{stream}_{condition}"
            params = defaults.copy()
            params["PATHS"] = [str(Path(campaign["PATH_DATASET"]) / stream)]
            params["USE_DETECTOR"] = condition != "supervised"
            params["DETECTOR_SIMULATOR_PRECISION"] = precision
            params["DETECTOR_SIMULATOR_RECALL"] = recall
            params["LOAD_DETECTIONS"] = row["load_detections"]
            params["RNN_LEARNERS"] = row["rnn_learners"]
            params["STREAM_LEARNERS"] = row["stream_learners"]
            params["JOB_NAME"] = name
            job_file = directory / f"{name}.json"
            with job_file.open("x", encoding="utf-8") as file:
                json.dump(params, file, indent=2)
            jobs.append((priority, number, job_file))

    return [job_file for _, _, job_file in sorted(jobs)]


def write_status(path, pending, active, completed, failed):
    """Publish a readable snapshot of the queue without exposing partial writes."""
    sections = (
        ("RUNNING", [f"{name} (PID {process.pid})" for process, name in active.items()]),
        ("COMPLETED", completed),
        ("FAILED", failed),
        ("TODO", [job_file.stem for job_file in pending]),
    )
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            for title, names in sections:
                print(f"{title} ({len(names)})", file=file)
                for name in names:
                    print(f"  {name}", file=file)
                print(file=file)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def run_jobs(module, job_files, output, workers=4, log_name="run.log"):
    """Run parameter files in order and collect queue and process logs."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if Path(log_name).name != log_name or log_name == "..":
        raise ValueError("OUTPUT_FILE must be a filename")
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.txt"

    with (output / "queue.log").open("a", buffering=1) as queue_log:
        pending = deque(job_files)
        active = {}
        completed = []
        failed = []
        print(f"Starting {len(pending)} jobs with {workers} slots", file=queue_log)
        write_status(status, pending, active, completed, failed)
        try:
            while pending or active:
                while pending and len(active) < workers:
                    job_file = pending[0]
                    log_path = output / job_file.stem / log_name
                    log_path.parent.mkdir(exist_ok=True)
                    command = [sys.executable, "-u", "-m", module, "--config", str(job_file)]
                    with log_path.open("a") as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    pending.popleft()
                    active[process] = job_file.stem
                    write_status(status, pending, active, completed, failed)
                    print(
                        f"Started {job_file.stem} (PID {process.pid}) - {log_path}"
                        f" | {len(active)} active, {len(pending)} queued",
                        file=queue_log,
                    )

                for process, name in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        del active[process]
                        if code == 0:
                            completed.append(name)
                        else:
                            failed.append(f"{name} (exit {code})")
                        write_status(status, pending, active, completed, failed)
                        print(
                            f"Finished {name} (exit {code})"
                            f" | {len(active)} active, {len(pending)} queued",
                            file=queue_log,
                        )
                if active:
                    time.sleep(0.2)
        finally:
            for process in active:
                if process.poll() is None:
                    process.terminate()
            for process, name in active.items():
                code = process.wait()
                if code == 0:
                    completed.append(name)
                else:
                    failed.append(f"{name} (exit {code})")
            if active:
                active.clear()
                write_status(status, pending, active, completed, failed)
        print(f"Queue complete, {len(failed)} failed", file=queue_log)
    return int(bool(failed))


def run(campaign_file, workers=4):
    """Expand a campaign into temporary files and run the evaluation jobs."""
    with open(campaign_file, encoding="utf-8") as file:
        campaign = json.load(file)
    results = Path(campaign.get("PATH_PERFORMANCE", PARAMETERS["PATH_PERFORMANCE"]))
    output = Path("outputs") / (results.name or Path(campaign_file).stem)
    with TemporaryDirectory() as directory:
        return run_jobs(
            "evaluation.test", create_jobs(campaign, Path(directory)), output,
            workers, campaign.get("OUTPUT_FILE") or "run.log",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Campaign JSON file")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sys.exit(run(args.config, args.workers))
