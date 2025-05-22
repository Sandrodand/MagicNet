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
* sine_rw10_mode5: SRWM
* weather: Weather

<ins>Train or test:</ins>
* train: The data stream contains the data points for the prequential evaluation.
* test: The data stream contains the data points of the test sets for the CL evaluation. Each concept (task column) is represented by 2k data points.

#### models
- models/cpnn: It contains the python modules implementing cPNNs:
    - models/cpnn/cpnn.py: the cPNN class implements cPNN's architecture.
    - models/cpnn/cpnn_columns.py: the cPNNColumns class implements the manager of cPNN's columns.
    - models/cpnn/inference_cpnn.py: the InferenceCPNN class implements the manager of cPNN ensemble mechanism of CL evaluation.
- models/crnn: It contains the python modules implementing cLSTM and cGRU. 
- models/magic: It contains the python modules implementing MAGIC Net.
  - models/magic/magic_net: the MagicNet class implements MAGIC Net's architecture.
  - models/cpnn/manager.py: the MagicManager class implements the manager of MAGIC Net's masks and models.
  - models/cpnn/piggyback_cgru.py: the PiggyBackGRU class implements the GRU model with masks.
  - models/cpnn/piggyback_layer.py: the different classes applies masks to the GRU model's layers.
- models/sml/temporally_augmented_classifier.py: The class TemporallyAugmentedClassifier implements temporal augmentation given a model.
### evaluation
It contains the python modules to implement the prequential evaluation used for the experiments.
#### data_gen
It contains the python modules implementing the data stream generator.
#### detectors
detectors/detector_simulator.py contains the class DetectorSimulator that simulates a concept drift detector given a data stream and target precision and recall values.

## 3) Evaluation
#### evaluation/test.py
It runs the prequential evaluation and CL evaluation using the specified configurations. Change the variables in the code for different settings (see the code's comments for the details).

Run it with the command `python -m evaluation.test`.

The execution stores the pickle files containing the results in the folder specified by the variable `PATH_PERFORMANCE`. For the details about the pickle files, see the documentation in **evaluation/prequential_evaluation.py** and **evaluation/cl_evaluation.py**.
