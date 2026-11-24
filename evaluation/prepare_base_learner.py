"""Save initial recurrent and linear weights used by the campaign."""

import json
from pathlib import Path

import pandas as pd

from evaluation.default_parameters import (
    set_batch_size, set_hidden_size, set_output_size, set_seq_len,
)
from evaluation.parameter_config import Config
from evaluation.test import PARAMETERS


CAMPAIGNS = Path(__file__).with_name("campaigns")


def campaign_streams():
    """Yield the training streams and detector conditions in the campaign files."""
    for filename in sorted(CAMPAIGNS.glob("*.json")):
        with filename.open(encoding="utf-8") as file:
            campaign = json.load(file)
        rows = campaign.pop("CAMPAIGN")
        params = PARAMETERS.copy()
        params.update(campaign)
        for row in rows:
            numbers = row["CONFIGURATIONS"]
            if isinstance(numbers, dict):
                first, last = numbers["RANGE"]
                numbers = range(first, last + 1)
            for number in numbers:
                name = f"{row['DATASET']}_{number}conf"
                train_name = name + "_train" if params["DO_CL"] else name
                yield params, row["DETECTOR"], name, Path(params["PATH_DATASET"]) / train_name


def prepare_base_learner(path, destination, rnn, params):
    """Initialize one base learner and save its weights and dimension metadata."""
    columns = pd.read_csv(f"{path}.csv", nrows=0).columns
    num_features = len(columns.drop(["task", "target", "batch"], errors="ignore"))
    dataset = path.name.lower()
    seq_len = params["SEQ_LEN_PARAM"]
    batch_size = params["BATCH_SIZE_PARAM"]
    output_size = params["NUM_CLASSES"]
    hidden_size = params["HIDDEN_SIZE_PARAM"][rnn]

    cfg = Config(base_rnn=rnn)
    cfg.set_params(
        num_features=num_features,
        seq_len=set_seq_len(dataset) if seq_len is None else seq_len,
        batch_size=set_batch_size(dataset) if batch_size is None else batch_size,
        output_size=set_output_size(dataset) if output_size is None else output_size,
        hidden_size={rnn: set_hidden_size(dataset, rnn) if hidden_size is None else hidden_size},
        save_column_freq=None,
        drift_delay=None,
    )
    cfg.prepare_base_learners(destination.parent.parent, destination.stem, load=False)
    print(f"Saved {destination}")


def main():
    """Save missing base learners one at a time."""
    for params, _, _, path in campaign_streams():
        root = Path("base_learner") / params["PATH_BASE_LEARNER"]
        for rnn in params["BASE_RNN"]:
            destination = root / rnn / f"{path.name}.pt"
            if not destination.exists():
                prepare_base_learner(path, destination, rnn, params)


if __name__ == "__main__":
    main()
