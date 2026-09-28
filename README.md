# MAGIC Net
This repository contains the code used for the experimentation shown in the paper.

Paper: Federico Giannini, Sandro D'andrea, Emanuele Della Valle: **Don't Look Back in Anger: MAGIC Net for Streaming Continual Learning with Temporal Dependence**. IEEE Big Data 2025: 1396-1403
- [Post proceeding version](https://ieeexplore.ieee.org/document/11401614)
- [Preprint version](https://arxiv.org/abs/2603.08600)

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
- models/cpnn: It contains the python modules implementing cPNN and DYNcPNN.
- models/crnn: It contains the python modules implementing cLSTM and cGRU. 
- models/magic: It contains the python modules implementing MAGIC Net.
* **`models/magic/`**: It contains the Python modules implementing MAGIC Net.
  * **`models/magic/magic_net.py`**: The `MagicNet` class implements MAGIC Net's architecture.
  * **`models/magic/manager.py`**: The `MagicManager` class implements the manager of MAGIC Net's masks and models.
  * **`models/magic/piggyback_cgru.py`**: The `PiggyBackGRU` class implements the GRU model with masks.
  * **`models/magic/piggyback_layers.py`**: The different classes apply masks to the GRU model's layers.
  * **`models/magic/activations.py`**: It contains custom activation and thresholding functions (e.g., `CappedSigmoid`, `Binarizer`, `Ternarizer`) used to modulate the continuous masks.
  * **`models/magic/cgru_from_masks.py`**: The `cGRUFromMasks` class reconstructs a static, standard GRU model by applying the learned masks to the frozen base weights.
  * **`models/magic/inference_magic.py`**: The `InferenceMagicNet` class implements a wrapper to perform continual inference by dynamically evaluating the ensemble of historical masks.
  * **`models/magic/inference_magic_fix.py`**: An updated version of the inference wrapper, which also includes the `RollingCohenKappa` utility for evaluating metrics over a rolling window.
- models/sml/temporally_augmented_classifier.py: The class TemporallyAugmentedClassifier implements temporal augmentation given a model.
### evaluation
It contains the python modules to implement the prequential evaluation used for the experiments.
#### detectors
detectors/detector_simulator.py contains the class DetectorSimulator that simulates a concept drift detector given a data stream and target precision and recall values.

## 3) Running the experiments
#### evaluation/test.py
It runs the prequential evaluation and CL evaluation using the specified configurations. Change the variables in the code for different settings (see the code's comments for the details).

Run it with the command `python -m evaluation.test`.

The execution stores the pickle files containing the results in the folder specified by the variable `PATH_PERFORMANCE`. For the details about the pickle files, see the documentation in **evaluation/prequential_evaluation.py** and **evaluation/cl_evaluation.py**.