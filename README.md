# MAGIC Net

Online and continual-learning experiments for MAGIC Net, its ablations and DYNcPNN,
with GRU and LSTM backbones.

## Installation

Use Python 3.12 and the pinned environment on macOS Intel or Linux:

```bash
conda env create -f environment.yml
conda activate magicnet
python -m pip check
```

`environment.yml` installs PyTorch CPU from conda-forge, which provides recent
builds for Intel Macs. The remaining dependencies are pinned in `requirements.txt`.
Install through the environment file first: PyPI does not provide these recent
PyTorch builds for macOS Intel. Keep the same package versions across machines.

If an older Conda on macOS incorrectly detects the system as macOS 10.16, use
`SYSTEM_VERSION_COMPAT=0 conda env create -f environment.yml`.

## Data and structure

Download [datasets.zip](https://drive.google.com/file/d/12MjLQhkL-EAS1dd0RlxP0bJ5rvmHG6j5/view?usp=sharing)
and extract it in the project root. Streams are named
`datasets/<dataset>_<configuration>conf_<train_or_test>.csv`:
`air_quality` (AirQuality), `energy` (PowerConsumption), and `weather` (Weather).
Training files provide the online stream; test files contain 2,000 points per
concept for CL evaluation. The `task` column identifies supervised concepts.

| Directory | Contents |
| --- | --- |
| `models/crnn/` | cGRU and cLSTM base learners. |
| `models/cpnn/` | cPNN and DYNcPNN. |
| `models/magic/` | MAGIC Net, masked recurrent layers and CL inference. |
| `models/mh_crnn/` | MultiHeadCRNN ablations and CL inference. |
| `models/sml/` | Temporal augmentation for stream learners. |
| `detectors/` | Sentinel and simulated detectors. |
| `evaluation/` | Evaluation, campaign launcher and input preparation. |

## Running experiments

Run from the project root:

```bash
python -m evaluation.test
python -m evaluation.test --config path/to/job.json
```

The first command uses `PARAMETERS` in `evaluation/test.py`; the second reads a
complete job JSON with the same keys (`null` represents Python `None`). Campaign
JSON files are passed to `evaluation.run_campaign`, as described below.

## Parameters

The defaults below refer to the manual runner.

#### Input, evaluation, and output

| Parameter | Default | What it controls |
| --- | --- | --- |
| `MODE` | `"local"` | `"local"` displays progress in the terminal. `"aws"` prints progress on separate lines and, for manual runs, redirects stdout to a text file under `outputs/`. The campaign launcher captures stdout and stderr in the job log. |
| `PATHS` | Weather configurations 1–10 under `datasets/` | List of stream paths without the CSV extension or the train and test suffixes. For example, use `datasets/weather_3conf`. The launcher sets a single stream path per job. |
| `PATH_PERFORMANCE` | `""` | Results directory. Relative paths are placed under `performance/`. Absolute paths are used directly. Each stream or campaign job gets a subdirectory. |
| `DO_CL` | `True` | Run continual-learning evaluation after each prequential iteration. With `True`, the runner reads `<path>_train.csv` and `<path>_test.csv`. With `False`, it reads `<path>.csv` for prequential evaluation. |
| `ITERATIONS` | `1` | Number of prequential evaluation repetitions per stream. Each repetition starts with fresh learner instances. |
| `WRITE_CHECKPOINTS` | `True` | Write the evaluation checkpoints to disk in `checkpoints<suffix>_it<iteration>.pkl`. CL evaluation can still use its in-memory checkpoints when disk writing is disabled. |
| `PREPROCESS` | `None` | Optional Python object for prequential feature preprocessing. It must provide `preprocess(x)` and `reset()`. Use `null` in a JSON job file. |
| `DELAY_PARAM` | `None` | Number of stream points between receiving a feature vector and receiving its label. Use `0` for immediate labels or `None` for the dataset default in `set_delay()`. |
| `CALLBACK_FUNC` | `None` | Optional Python callback invoked at the end of each prequential iteration when `DO_CL=False`. With `DO_CL=True`, the runner uses the CL evaluator's callback. Use `null` in a JSON job file. |
| `OUTPUT_FILE` | `None` | Manual AWS runs use `outputs/<OUTPUT_FILE>.txt`, deriving the name from the first stream and simulated detector rates when unset. Campaign logs use the generated job name. |
| `SUFFIX` | `""` | Optional string appended to result filenames. For example, `"_trial2"` produces `performance_trial2.pkl`. Learner names include the recurrent architecture, such as `MAGIC_Net_gru`. |
| `JOB_NAME` | `None` | Name of the job's results subdirectory. The launcher sets it to the name shared by the results and log directories. Use `None` for manual runs. |

#### Drift detection

| Parameter | Default | What it controls |
| --- | --- | --- |
| `USE_DETECTOR` | `True` | Use simulated or Sentinel detections. Set `False` to use the supervised drift boundaries from the CSV's `task` column in a manual run. |
| `DETECTOR_SIMULATOR_PRECISION` | `0.7` | Target fraction of simulated alarms that correspond to a true drift. Use a fraction such as `0.75` for 75%. The simulator rounds to an integer number of alarms. |
| `DETECTOR_SIMULATOR_RECALL` | `1` | Target fraction of true drifts that trigger a simulated alarm. The simulator rounds to an integer number of detected drifts. With `USE_DETECTOR=True`, setting both precision and recall to `None` selects Sentinel. |
| `DETECTOR_SIMULATOR_MIN_DELAY` | `0` | Inclusive lower bound on the number of points after a true drift when the simulator can place a matching alarm. |
| `DETECTOR_SIMULATOR_MAX_DELAY` | `1000` | Exclusive upper bound on that simulated delay. The default interval is 0–999 points, including for 100/100. The runner also uses this parameter to configure rollback for simulated conditions. |
| `LOAD_DETECTIONS` | `False` | Replay saved alarm indices when enabled. With `False`, the runner generates simulated alarms, runs Sentinel live, or uses supervised boundaries according to the selected detector. Saved Sentinel alarms are applied at their recorded index after prediction and before training. |
| `PATH_LOAD_DETECTIONS` | `None` | Root of the saved detections. With a separate root, the runner reads `<root>/<stream-or-job-name>/drifts.pkl`. With `None`, it uses the results root. When a campaign shares the input and results root, its input file is `prepared_drifts.pkl` inside the job directory. The pickle contains detection lists by iteration and the runner reuses the first list. |

#### Base learners and model dimensions

| Parameter | Default | What it controls |
| --- | --- | --- |
| `SAVE_BASE_LEARNER` | `True` | Save recurrent and initial linear-head weights with architecture/dimension metadata in a `.pt` file, before training. Existing files are reused when loading is enabled. |
| `PATH_BASE_LEARNER` | `""` | Root directory for initial weights. Relative paths are placed under `base_learner/`, e.g. `base_learner/gru/weather_3conf_train.pt`. |
| `UPLOAD_BASE_LEARNER` | `True` | Reconstruct a fresh base learner from compatible saved weights. If the file is missing, initialize new weights. Set `False` to initialize new weights even when an archive exists. |
| `BASE_RNN` | `["gru"]` | Recurrent architectures to evaluate. Use `["gru"]`, `["lstm"]`, or `["gru", "lstm"]`. Each architecture runs the selected recurrent methods on the same stream and detector signal. |
| `SEQ_LEN_PARAM` | `None` | Number of points in the recurrent input sequence. Use `None` for the dataset default. MAGIC Net and MultiHeadCRNN require a sequence length of at least 2. |
| `BATCH_SIZE_PARAM` | `None` | Number of labeled points per recurrent training minibatch. The evaluator also uses this value to convert rolling metric windows from minibatches to points. `None` uses the dataset default. |
| `HIDDEN_SIZE_PARAM` | `{"gru": None, "lstm": None}` | Initial recurrent hidden size for each architecture. A `None` entry uses the dataset and architecture default. |
| `MAX_OLD_LABELS` | `20` | Maximum history length used for temporal augmentation. The runner uses `min(sequence_length - 1, MAX_OLD_LABELS)` for the temporal order. |
| `NUM_CLASSES` | `None` | Number of outputs in the classification head. `None` uses the dataset default. |

#### Drift ensembles

Each dictionary in this table must contain an entry for every architecture
listed in `BASE_RNN`.

| Parameter | Default | What it controls |
| --- | --- | --- |
| `ENSEMBLE_BATCHES` | `{"gru": 50, "lstm": 50}` | Length of the candidate comparison after a detection in MAGIC Net, MultiHeadCRNN, and DYNcPNN, expressed in minibatches. |
| `SELECTION_BATCHES` | `{"gru": 40, "lstm": 40}` | Rolling Cohen's Kappa window for selecting a MAGIC Net or MultiHeadCRNN candidate during the comparison and at its end, expressed in minibatches. |
| `DYNCPNN_SELECTION_BATCHES` | `{"gru": 40, "lstm": 40}` | Rolling Cohen's Kappa window for selecting a DYNcPNN candidate during the comparison and at its end, expressed in minibatches. |
| `HIDDEN_MULT` | `{"gru": None, "lstm": None}` | Number of recurrent units added by an expansion in MAGIC Net or MultiHeadCRNN. A `None` entry uses half the configured initial hidden size, rounded down. |
| `ENSEMBLE_TH` | `{"gru": 1.05, "lstm": 1.05}` | Multiplier for accepting an expansion in MAGIC Net or MultiHeadCRNN. The expansion's Kappa must strictly exceed both the best non-expansion Kappa and that score multiplied by this value. |

#### Learners

| Parameter | Default | What it controls |
| --- | --- | --- |
| `RNN_LEARNERS` | `["MAGIC_Net"]` | Recurrent methods instantiated for each architecture in `BASE_RNN`. Available names are `cRNN`, `cPNN`, `DYNcPNN`, `MAGIC_Net`, `MAGIC_Net_sh`, `MAGIC_Net_binary`, `MAGIC_Net_no_ignore`, `MAGIC_Net_no_expand`, `MultiHeadCRNN_frozen`, and `MultiHeadCRNN_learnable`. |
| `STREAM_LEARNERS` | `[]` | Optional stream baselines instantiated once per job. Available names are `ARF`, `ARF_TA`, `HAT`, and `HAT_TA`. |

## Campaigns

Store campaign JSON files in `evaluation/campaigns/`. Top-level settings override
the manual defaults above. Campaign-specific keys are:

| Key | Meaning |
| --- | --- |
| `PATH_DATASET` | Dataset directory used to construct stream paths. |
| `CAMPAIGN` | List of experiment definitions with the following keys. |
| `DATASET` | Dataset name, such as `weather`. |
| `CONFIGURATIONS` | Explicit list, e.g. `[1, 3, 5]`, or inclusive range `{"RANGE": [1, 20]}`. |
| `DETECTOR` | `[precision, recall]` for simulation, `"sentinel"` for Sentinel, or `"supervised"` / `false` for CSV task boundaries. |
| `LOAD_DETECTIONS` | Per-definition switch to replay saved detections instead of running the selected detector live. |
| `RNN_LEARNERS` / `STREAM_LEARNERS` | Methods to evaluate, using the names listed above. |

Prepare shared inputs, then launch a campaign:

```bash
python -m evaluation.prepare_base_learner
python -m evaluation.prepare_detections
python -m evaluation.run_campaign --config evaluation/campaigns/weather_gru.json --workers 4
```

Preparation reads every campaign JSON and skips existing files. Base learners
are prepared sequentially; detections use four workers and are shared across
GRU/LSTM. `--workers` controls concurrent experiment jobs (default `4`). Jobs run
100/100 first, then Sentinel, then other detectors by descending precision and
recall, then supervised; configurations are ordered numerically per condition.

For a campaign named `weather_gru.json`:

```text
outputs/weather_gru/queue.log                 # job starts and exits
outputs/weather_gru/status.txt                # running, completed, failed, todo
outputs/weather_gru/processes/<job>.log        # individual job output
performance/<PATH_PERFORMANCE>/<job>/         # results, for relative paths
```

Job names follow `<dataset>_<conf>conf_<detector>`, e.g.
`weather_3conf_75prec_100rec`. Status is refreshed at each job transition.
Detection preparation uses the same log layout under `outputs/prepare_detections/`;
Sentinel reports `processed/total` every 100 points.

```bash
tail -F outputs/weather_gru/processes/weather_3conf_75prec_100rec.log
```

## Backbones and ablations

To compare architectures on the same stream and detections:

```python
PARAMETERS["BASE_RNN"] = ["gru", "lstm"]
PARAMETERS["HIDDEN_SIZE_PARAM"] = {"gru": None, "lstm": None}
PARAMETERS["RNN_LEARNERS"] = ["MAGIC_Net", "MultiHeadCRNN_frozen", "DYNcPNN"]
```

Within each architecture, methods receive independent copies of the same base
learner. Archives use `<PATH_BASE_LEARNER>/<gru_or_lstm>/<dataset>.pt` and contain
only weights and metadata. Loading validates architecture and dimensions, then
constructs a new learner with a fresh Adam optimizer, metrics and buffers while
preserving the experiment's random-generator state. Old `.pkl` base learners are
not read; regenerate initializations with `python -m evaluation.prepare_base_learner`.
All methods share the initial weights and optimizer settings and follow the
same training trajectory before the first detection. Results distinguish architectures,
e.g. `MAGIC_Net_gru` and `MAGIC_Net_lstm`. Direct `Config` factories accept
`rnn="gru"` or `rnn="lstm"`; omission is allowed only with one architecture.

MultiHeadCRNN compares `HeadOnly` (new head), optional `Ignore` (current model),
and optional `Expand` (larger hidden size and new head). The frozen variant uses
rollback for HeadOnly/Expand and freezes existing recurrent weights; expansion
leaves new connections trainable. The learnable variant starts from the current
model and keeps recurrent weights trainable. Historical heads remain usable
after further expansions.

## Direct API and checkpoints

Additional model and inference options:

| Parameter | Meaning |
| --- | --- |
| `base_learner_weights` | Reuse a cPNN or cGRULinear/cLSTMLinear column through independent copies, including its dimensions and initial head. |
| `model_class` | Without a supplied base learner, MAGIC defaults to GRU; use `PiggyBackLSTM` from `models.magic.piggyback_clstm` for LSTM. |
| `seq_len` | Temporal window length; must be at least 2. |
| `reset_data_points` | Clear temporal context at an accepted drift when `True`; otherwise preserve the latest available `seq_len - 1` training observations when discarding the partial minibatch. |
| `timestamp` | Match each prediction to its label, including delayed labels; use the same point's timestamp in `predict_one` and `learn_one`. |
| `initial_task` | Evaluator's initial dataset task label; internal MAGIC task IDs independently start at 1. |
| `name` | Prefix for model diagnostics; defaults to `MAGIC_Net`, `DYNcPNN` or `MultiHeadCRNN`. Factories append variant and architecture. |
| `freeze_backbone` | MultiHeadCRNN freeze policy: `True` uses rollback and freezes existing recurrent parameters for HeadOnly/Expand; `False` keeps them trainable without rollback. |
| `ignore_option` | Include the Ignore candidate. |
| `expand_option` | Include hidden-size expansion; HeadOnly remains available in MultiHeadCRNN. |
| `minimum_k_difference` | Minimum additive Kappa advantage for retaining the pre-drift MAGIC model after Ignore (default `0.04`, with a strictly positive advantage). |
| `data_points_previous_concept` | Maximum previous-concept sample size for the forgetting check (default `1000`); collection continues beyond ensembling. |
| `ensemble_data_points` | CL label updates before retaining the best inference candidate (default `500`); `-1` keeps all candidates throughout inference. |
| `rolling_window` | CL candidate Kappa window in observations; `None` uses cumulative Kappa. |
| `recurrent_mask_weights` / `linear_mask_weights` | Initial mask values for the generic recurrent / linear masked layers. |

The evaluator manages timestamps automatically. For direct streaming use:

```python
prediction = model.predict_one(x_t, timestamp=t)
# When the label arrives:
model.learn_one(x_t, y_t, timestamp=t)
```

MAGIC and MultiHeadCRNN expose `learn_one`; minibatch training is internal.
Mask-Last initializes newly frozen sigmoid masks to `0.9`; historical mask
padding disables connections added by later expansions. After Ignore, the
rollback task is retained only when the old-concept comparison passes the
retention threshold. A full backbone is archived only when needed to preserve
trainable weights; the learnable MultiHeadCRNN ablation retains heads only.

Use `pickle.dump(model, file)` / `pickle.load(file)` for complete checkpoints,
including candidates, optimizers and pending labels. CL resolves a pending
comparison on a copy and resets temporal context; warm-up predictions are class
zero. `get_state_dict()` returns `None` during comparison and cannot replace a
full checkpoint. `force_decision()` does nothing without a pending comparison.

Results are saved under `PATH_PERFORMANCE`; see `evaluation/prequential_evaluation.py`
and `evaluation/cl_evaluation.py` for pickle schemas. Execution time is recorded
only for prequential training, in microseconds; CL has no time metric.
