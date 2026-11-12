# MAGIC Net
This repository contains the code used for the experimentation shown in the paper.

## 1) Installation
execute:

`conda create -n env python=3.8`

`conda activate env`

`pip install -r requirements.txt`

## 2) Project structure
The project is composed of the following directories.
#### datasets
Download `datasets.zip` from [here](https://drive.google.com/file/d/12MjLQhkL-EAS1dd0RlxP0bJ5rvmHG6j5/view?usp=sharing).

Extract it in the main folder of the project.

It contains the generated data streams.
Each file's name has the following structure: **\<data_source\>\_\<id_configuration\>conf\_\<train_or_test\>.csv**.

<ins>Data sources:</ins>
* air_quality: AirQuality.
* energy: PowerConsumption.
* weather: Weather

<ins>Train or test:</ins>
* train: The data stream contains the data points for the prequential evaluation.
* test: The data stream contains the data points of the test sets for the CL evaluation. Each concept (task column) is represented by 2k data points.

#### models
- `models/cpnn/`: It contains the python modules implementing cPNN and DYNcPNN.
- `models/crnn/`: It contains the python modules implementing cLSTM and cGRU. 
- `models/magic/`: It contains the python modules implementing MAGIC Net.
- **`models/magic/`**: It contains the Python modules implementing MAGIC Net.
  * **`models/magic/magic_net.py`**: The `MagicNet` class implements MAGIC Net's architecture.
  * **`models/magic/manager.py`**: The `MagicManager` class implements the manager of MAGIC Net's masks and models.
  * **`models/magic/piggyback_rnn.py`**: `PiggyBackRNN` provides the shared recurrent masking, heads and expansion lifecycle.
  * **`models/magic/piggyback_cgru.py`**: The `PiggyBackGRU` class implements the GRU model with masks.
  * **`models/magic/piggyback_clstm.py`**: `PiggyBackLSTM` implements the masked LSTM. GRU and LSTM both inherit directly from `PiggyBackRNN`.
  * **`models/magic/piggyback_layers.py`**: The recurrent and linear layers apply the weight masks.
  * **`models/magic/activations.py`**: It contains custom activation and thresholding functions (e.g., `CappedSigmoid`, `Binarizer`, `Ternarizer`) used to modulate the continuous masks.
  * **`models/magic/cgru_from_masks.py`**: The `cGRUFromMasks` class reconstructs a static, standard GRU model by applying the learned masks to the frozen base weights.
  * **`models/magic/inference_magic.py`**: The `InferenceMagicNet` class implements a wrapper to perform continual inference by dynamically evaluating the ensemble of historical masks.
  * **`models/magic/inference_magic_fix.py`**: An updated version of the inference wrapper, which also includes the `RollingCohenKappa` utility for evaluating metrics over a rolling window.
- `models/sml/temporally_augmented_classifier.py`: The class TemporallyAugmentedClassifier implements temporal augmentation given a model.
### evaluation
It contains the python modules to implement the prequential evaluation used for the experiments.
#### detectors
detectors/detector_simulator.py contains the class DetectorSimulator that simulates a concept drift detector given a data stream and target precision and recall values.

## 3) Running the experiments
### Running an evaluation

Run `python -m evaluation.test` from the project root to use the
`PARAMETERS` dictionary in [evaluation/test.py](evaluation/test.py).
To use a complete JSON parameter file, run
`python -m evaluation.test --config path/to/job.json`. The JSON uses the
same keys as `PARAMETERS` and represents Python `None` as JSON `null`.

### Evaluation parameters

The defaults below refer to the manual runner.

#### Input, evaluation, and output

| Parameter | Default | What it controls |
| --- | --- | --- |
| `MODE` | `"local"` | `"local"` displays progress in the terminal. `"aws"` prints progress on separate lines and, for manual runs, redirects stdout to a text file under `outputs/`. The campaign launcher captures stdout and stderr in the job log. |
| `PATHS` | Weather configurations 1–10 under `datasets/` | List of stream paths without the CSV extension or the train and test suffixes. For example, use `datasets/weather_3conf`. The launcher sets a single stream path per job. |
| `PATH_PERFORMANCE` | `"debug_ensemble"` | Results directory. Relative paths are placed under `performance/`. Absolute paths are used directly. Each stream or campaign job gets a subdirectory. |
| `DO_CL` | `True` | Run continual-learning evaluation after each prequential iteration. With `True`, the runner reads `<path>_train.csv` and `<path>_test.csv`. With `False`, it reads `<path>.csv` for prequential evaluation. |
| `ITERATIONS` | `1` | Number of prequential evaluation repetitions per stream. Each repetition starts with fresh learner instances. |
| `WRITE_CHECKPOINTS` | `True` | Write the evaluation checkpoints to disk in `checkpoints<suffix>_it<iteration>.pkl`. CL evaluation can still use its in-memory checkpoints when disk writing is disabled. |
| `PREPROCESS` | `None` | Optional Python object for prequential feature preprocessing. It must provide `preprocess(x)` and `reset()`. Use `null` in a JSON job file. |
| `DELAY_PARAM` | `None` | Number of stream points between receiving a feature vector and receiving its label. Use `0` for immediate labels or `None` for the dataset default in `set_delay()`. |
| `CALLBACK_FUNC` | `None` | Optional Python callback invoked at the end of each prequential iteration when `DO_CL=False`. With `DO_CL=True`, the runner uses the CL evaluator's callback. Use `null` in a JSON job file. |
| `OUTPUT_FILE` | `None` | Manual AWS runs use `outputs/<OUTPUT_FILE>.txt`, deriving the name from the first stream and simulated detector rates when unset. Campaigns use this as the log filename inside each job directory and default to `run.log`. |
| `suffix` | `""` | Optional string appended to result filenames. For example, `"_trial2"` produces `performance_trial2.pkl`. Learner names include the recurrent architecture, such as `MAGIC_Net_gru`. |
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
| `SAVE_BASE_LEARNER` | `True` | Save the initial recurrent model for each architecture and stream, before evaluation training. With loading enabled, an existing architecture-specific archive is reused. |
| `PATH_BASE_LEARNER` | `""` | Root directory for initial model archives. Relative paths are placed under `base_learner/`. For example, the default GRU archive for a CL run on Weather configuration 3 is `base_learner/gru/weather_3conf_train.pkl`. |
| `UPLOAD_BASE_LEARNER` | `True` | Try to load a compatible local base learner archive. If no archive is found, initialize a fresh model. Set `False` to start from a fresh initialization even when an archive exists. |
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

### Running a campaign

Prepare the shared inputs from the project root.

```bash
python -m evaluation.prepare_base_learner
python -m evaluation.prepare_detections
```

Both scripts read all JSON files in `evaluation/campaigns/` and reuse
existing output files. Base learners are saved sequentially per stream and
architecture. Detection files are prepared with up to four concurrent jobs
and can be shared between GRU and LSTM.

Follow `outputs/prepare_detections/queue.log` to see which detection jobs are
running. Each detection job has its own log. Sentinel reports
`processed/total` every 100 stream points and at the end of the stream.

```bash
tail -F outputs/prepare_detections/weather_3conf_sentinel/run.log
```

```bash
python -m evaluation.run_campaign --config evaluation/campaigns/weather_gru.json --workers 4
```

The campaign file overrides the manual defaults with common settings at its
top level. `PATH_DATASET` is the dataset directory. Each row in `campaign`
specifies `dataset`, `configurations`, `detector`, `load_detections`,
`rnn_learners`, and `stream_learners`. Use an explicit list such as
`[1, 3, 5]` or an inclusive range such as `{"range": [1, 20]}` for
`configurations`. A detector pair such as `[0.75, 1.0]` selects simulated
precision and recall. Use `"sentinel"` for Sentinel and `"supervised"` or
`false` for supervised boundaries.

The launcher writes a temporary parameter file for each job and passes it to
`evaluation.test --config`. It starts 100/100 first, then Sentinel, then
the other simulated conditions by descending precision and recall, followed
by supervised conditions. Configurations are ordered numerically within each
condition. `--workers` limits the number of concurrent jobs and defaults to
four. The launcher removes the temporary files when the queue ends.

Follow `outputs/gru/queue.log` to see which jobs have started and finished.
Each job has its own `run.log` in a directory with the same name as its
results directory under `performance/gru/`, such as
`weather_3conf_75prec_100rec`.

`EvaluatePrequential` evaluates only `anytime_learners`, through `predict_one`
and `learn_one`. Batch learners and the periodic evaluation scenario have been
removed. The `_anytime` keys and output filenames are retained for CL evaluation
and existing result analysis. Its `batch_size` sets rolling metric windows.
Recurrent learners manage their training minibatches.

### Streaming API and delayed labels

MAGIC Net and `MultiHeadCrnn` expose `learn_one` for training. Mini-batch
training is internal (`_learn_many`); there is no public `learn_many` method.
Both require `seq_len >= 2`; shorter windows raise `ValueError` at construction.
MAGIC initializes its own linear head when no base learner is supplied. Imported
recurrent weights, linear weights and biases are copied, so training does not
modify a supplied base learner or another model initialized from that learner.
MAGIC and `MultiHeadCrnn` always start with internal task 1; `initial_task_id`
is no longer a constructor parameter. Dataset task labels remain independent
and are handled by the evaluator's `initial_task`. Calling `force_decision()`
when no drift comparison is pending leaves the model unchanged.
With delayed labels, pass the same point's timestamp to prediction and training:

```python
prediction = model.predict_one(x_t, timestamp=t)
# When the label for point t_labeled becomes available:
model.learn_one(x_labeled, y_labeled, timestamp=t_labeled)
```

The prequential evaluator passes these timestamps automatically. Immediate-label
usage without timestamps remains supported. During a drift comparison, candidate
predictions are stored by timestamp until their labels arrive; complete pickle
checkpoints also retain pending predictions. Labels for points that predate the
comparison or have no prediction during sequence warm-up are not scored.
The grace period counts labeled data points once, independently of mini-batch
training, starting after the point that completes the drift decision.

MAGIC supports both GRU and LSTM backbones with the same masking and drift
lifecycle. Existing experiment factories automatically select the architecture
of the shared base learner (selected through the `BASE_RNN` list). Passing a cPNN or
its `cGRULinear` / `cLSTMLinear` column to `base_learner_weights` reuses its weights,
biases, initial head and dimensions through independent copies. Without a base
learner, GRU remains the default; select LSTM explicitly with
`model_class=PiggyBackLSTM` from `models.magic.piggyback_clstm`.

LSTM expansion and historical masks preserve all four gate blocks and reset
both hidden and cell initial states to the new size. Single-head, multi-head,
sigmoid/binary masks, Mask-Last, conditional old-model retention and CL checkpoint
handling share the same implementation across the two architectures. Existing
GRU class paths remain available for loading previous pickle checkpoints.
`PiggyBackGRU` and `PiggyBackLSTM` are sibling subclasses of `PiggyBackRNN`;
all common model behavior lives in that base class.
The masked layers follow the same structure: `ElementWiseGRU` and
`ElementWiseLSTM` both extend `ElementWiseRNN`. The concrete layers define gate
order/count, initial states and recurrent computation: GRU uses reset/update/new
gates and a hidden state; LSTM uses input/forget/cell/output gates and hidden plus
cell states. Mask application, freezing and expansion remain shared and use the
declared gate count when copying parameter blocks. Historical mask padding uses
the same metadata. Generic APIs use `base_learner_weights`,
`recurrent_mask_weights` and `linear_mask_weights`. Previous constructor aliases
are not accepted, and old checkpoint attribute names are not migrated.

Sigmoid weight masks always use `CappedSigmoid`; the `cap_sigmoid` option has
been removed from constructors and experiment factories. Historical masks are
padded with effective zeros. Their reconstruction preserves exact zero and one,
so connections introduced by later expansions remain disabled for older tasks.
Binary masks retain their existing binarization behavior.

After `Ignore`, MAGIC compares the adapted candidate with the pre-drift model
retrieved through rollback on a sample of the preceding concept. The full old
backbone and head are archived in `forgotten_models` only if their Kappa advantage
is positive and reaches `minimum_k_difference` (default `0.04`). Both networks
are evaluated on independent copies with a fresh local temporal context; online
buffers, pending predictions and optimizers are unchanged. The concept sample
continues filling outside the ensemble phase up to `data_points_previous_concept`
(default `1000`). Historical heads, masks and single-head biases are retained
for CL reconstruction even when no full snapshot is archived. The same policy
applies to sigmoid and binary masks and to the unmasked ablation.

At a drift, Mask-Last freezes previously trainable weights and initializes their
sigmoid masks to an effective value of `0.9` (logit approximately `2.1972`). Masks
on already frozen weights retain their previous logits. This includes recurrent
bias masks and, in the single-head variant, linear weight masks. Binary masks
retain their logits and their straight-through estimator.

CL initialization resets the temporal context on the freshly reconstructed
task models, including models restored from the pool. Each test sequence starts
without training data in its window; repeated resets leave the source checkpoint
intact. During the first `seq_len - 1` points, the underlying models return
`None`, which the inference wrapper continues to represent as class zero.

### Comparing multiple recurrent backbones

Set these entries in `PARAMETERS` in `evaluation/test.py`:

```python
PARAMETERS["BASE_RNN"] = ["gru", "lstm"]
PARAMETERS["HIDDEN_SIZE_PARAM"] = {"gru": None, "lstm": None}
PARAMETERS["RNN_LEARNERS"] = ["MAGIC_Net", "MultiHeadCRNN_frozen", "DYNcPNN"]
```

`None` selects the dataset/architecture default. Sequence length, batch size,
preprocessing, label delay and detector remain common. All selected methods run
in one prequential/CL evaluation, with names such as `MAGIC_Net_gru` and
`MAGIC_Net_lstm`, receiving the same stream and detections.

Diagnostic messages use each model's `name`: defaults are `MAGIC_Net`, `DYNcPNN`
and `MultiHeadCRNN`. Experiment factories include the variant and architecture,
for example `MAGIC_Net_binary_sh_lstm: cancels the grace period`. Constructors
and factories accept a custom `name`; assigning `model.name` also updates MAGIC's
manager so subsequent messages use the new identifier.

`Config(base_rnn=PARAMETERS["BASE_RNN"])` stores `hidden_size`, `hidden_mult`,
`ensemble_batches`, `ensemble_th` and `base_learner` by RNN key. Factories accept
`rnn="gru"` or `rnn="lstm"`; omitting it is supported only for a single-RNN
configuration. `create_learners()` binds each factory explicitly to its RNN.
Base learners are initialized lazily after dataset parameters are resolved.
Within an architecture, every method receives independent copies of the same
initialization. Changing only ensemble settings preserves that initialization.

Archives are stored in `<PATH_BASE_LEARNER>/gru/<dataset>.pkl` and
`<PATH_BASE_LEARNER>/lstm/<dataset>.pkl`. Loading validates architecture, hidden
size, feature count, output classes, sequence length and batch size. A compatible
legacy flat archive is reused only for its own architecture and may be copied
into the corresponding subdirectory. Incompatible typed archives raise an error
instead of being silently replaced.

### MultiHeadCRNN ablations

Include `"MultiHeadCRNN_frozen"` or `"MultiHeadCRNN_learnable"` in
`RNN_LEARNERS` to compare these ablations for every selected architecture.
For direct construction of a frozen-backbone entry:

```python
from functools import partial

LearnerConfig(
    name="MultiHeadCRNN_frozen_lstm",
    model=partial(CFG.create_multihead_crnn_frozen, rnn="lstm"),
    drift=True,
    magic=True,
)
```

The factory reuses the initialized base learner shared by the other methods,
including its recurrent weights and initial linear head. Both GRU and LSTM
entries in `BASE_RNN` are supported through `cGRULinear` and `cLSTMLinear`.

The drift ensemble compares `Ignore`, `HeadOnly`, and `Expand` when all
candidates are enabled. `HeadOnly` adds a head at the current width.
`Expand` increases the recurrent hidden size and adds a head. The frozen
variant starts these two candidates from rollback and freezes the existing
backbone parameters. The learnable variant starts from the current model
and keeps the recurrent weights trainable. `ENSEMBLE_TH` applies to
`Expand` against the better of the available non-expansion candidates.

If `Ignore` wins, the manager uses the pre-drift checkpoint for the
old-concept Kappa comparison. Retention depends on that comparison and the
freeze policy. The learnable variant retains historical heads, and the frozen
variant can also retain a copy of a backbone that still had trainable weights.

CL evaluation automatically uses `InferenceMultiHeadCRNN`, which evaluates
every stored task with the appropriate backbone and head. Direct usage also
supports `ensemble_data_points=-1` to keep all candidates throughout inference,
and `rolling_window` for a rolling Cohen's Kappa.

Save and resume a complete training checkpoint with `pickle.dump(model, file)`
and `pickle.load(file)`, as in the existing evaluation pipeline. This includes
pending drift candidates, optimizers, metrics and the pre-drift snapshot. If CL
inference starts from a checkpoint taken during a drift comparison, it selects
the provisional winner on an independent copy, leaving training resumable.
`get_state_dict()` follows MAGIC's weights-only API and returns `None` during
a pending comparison; it is not a replacement for a complete pickle checkpoint.

Run the regression tests with `python -m pytest -q tests/test_head_only_magic.py`.

The execution stores the pickle files containing the results in the folder specified by the variable `PATH_PERFORMANCE`. For the details about the pickle files, see the documentation in **evaluation/prequential_evaluation.py** and **evaluation/cl_evaluation.py**.
