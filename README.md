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
  * **`models/magic/piggyback_cgru.py`**: The `PiggyBackGRU` class implements the GRU model with masks.
  * **`models/magic/piggyback_layers.py`**: The different classes apply masks to the GRU model's layers.
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
#### evaluation/test.py
It runs the prequential evaluation and CL evaluation using the specified configurations. Change the variables in the code for different settings (see the code's comments for the details).

Run it with the command `python -m evaluation.test`.

`EvaluatePrequential` evaluates only `anytime_learners`, through `predict_one`
and `learn_one`. Batch learners and the periodic evaluation scenario have been
removed. The `_anytime` keys and output filenames are retained for CL evaluation
and existing result analysis. Its `batch_size` still sets rolling metric windows
and output filenames; recurrent learners manage their training mini-batches.

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

Sigmoid weight masks always use `CappedSigmoid`; the `cap_sigmoid` option has
been removed from constructors and experiment factories. Historical masks are
padded with effective zeros. Their reconstruction preserves exact zero and one,
so connections introduced by later expansions remain disabled for older tasks.
Binary masks retain their existing binarization behavior.

CL initialization resets the temporal context on the freshly reconstructed
task models, including models restored from the pool. Each test sequence starts
without training data in its window; repeated resets leave the source checkpoint
intact. During the first `seq_len - 1` points, the underlying models return
`None`, which the inference wrapper continues to represent as class zero.

### Head-only ablation (no weight masks)

Add this entry to the experiment's `learners` list:

```python
LearnerConfig(
    name="MAGIC_Net_head_only",
    model=CFG.create_multihead_crnn,
    numeric=True,
    batch_learner=False,
    drift=True,
    magic=True,
)
```

The factory reuses the initialized base learner shared by the other methods,
including its recurrent weights and initial linear head. Both `BASE_RNN = "gru"`
and `BASE_RNN = "lstm"` are supported through `cGRULinear` and `cLSTMLinear`.

At every drift, `Ignore` continues training the current model, while `Expand`
retrieves the pre-drift checkpoint, freezes its backbone and trains a new head.
The backbone remains trainable until `Expand` is first selected; afterwards,
`Ignore` only fine tunes the current head. Expansion must outperform `Ignore`
and exceed its score multiplied by `ENSEMBLE_TH`; ties choose `Ignore`.
There is no increase in recurrent hidden size.

If `Ignore` wins, the full pre-drift model is stored in `forgotten_models`, using
MAGIC's delayed checkpoint retrieval through `restore_previous_state()` /
`get_previous_state()`. This retains the historical backbone as well as the head
when the backbone is still trainable. Checkpoint frequency, detector delay and
grace period follow the same behavior as MAGIC Net.

CL evaluation automatically uses `InferenceHeadOnlyMagicNet`, which evaluates
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
