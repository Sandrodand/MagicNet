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

## 3) Running the experiments
#### evaluation/test.py
It runs the prequential evaluation and CL evaluation using the specified configurations. Change the variables in the code for different settings (see the code's comments for the details).

Run it with the command `python -m evaluation.test`.

The execution stores the pickle files containing the results in the folder specified by the variable `PATH_PERFORMANCE`. For the details about the pickle files, see the documentation in **evaluation/prequential_evaluation.py** and **evaluation/cl_evaluation.py**.

## 4) Expected Results
### Prequential Evaluation (Cohen's Kappa)

**Note**: Mean ± std over 50 configurations. **Bold** indicates best models.

| Precision/Recall | Model     | AirQuality Start | AirQuality End | PowerCons. Start | PowerCons. End | SRW Start | SRW End | Weather Start | Weather End |
|------------------|-----------|------------------|----------------|------------------|----------------|-----------|---------|----------------|--------------|
| **100% / 100%**  | ARF       | 0.07±0.02        | 0.08±0.02      | 0.25±0.05        | 0.29±0.05      | 0.24±0.02 | 0.33±0.01 | 0.26±0.05      | 0.29±0.04    |
| 100% / 100       | ARF$_T$     | 0.21±0.05        | 0.26±0.04      | 0.22±0.06        | 0.28±0.06      | **0.73±0.00** | 0.73±0.00 | 0.41±0.06      | 0.44±0.06    |
| 100% / 100%      | cGRU      | 0.33±0.05        | 0.39±0.05      | 0.59±0.07        | 0.70±0.04      | 0.69±0.05 | **0.81±0.03** | 0.50±0.06      | 0.61±0.06    |
| 100% / 100%      | cPNN      | 0.35±0.03        | 0.35±0.03      | 0.66±0.02        | 0.71±0.02      | 0.61±0.05 | 0.68±0.04 | 0.52±0.05      | 0.56±0.05    |
| 100% / 100%      | MAGIC     | **0.42±0.03**    | **0.47±0.03**  | **0.68±0.04**    | **0.77±0.02**  | 0.71±0.04 | **0.82±0.03** | **0.56±0.04**  | **0.67±0.03**|
| **100% / 70%**   | ARF       | 0.07±0.02        | 0.08±0.02      | 0.24±0.06        | 0.28±0.06      | 0.24±0.02 | 0.33±0.01 | 0.27±0.06      | 0.29±0.05    |
| 100% / 70%       | ARF$_T$     | 0.20±0.05        | 0.26±0.05      | 0.21±0.06        | 0.27±0.07      | **0.73±0.00** | 0.73±0.00 | 0.42±0.07      | 0.45±0.06    |
| 100% / 70%       | cGRU      | 0.32±0.05        | 0.39±0.06      | 0.58±0.08        | 0.70±0.05      | 0.69±0.05 | **0.81±0.03** | 0.51±0.07      | 0.61±0.06    |
| 100% / 70%       | cPNN      | 0.34±0.03        | 0.35±0.03      | 0.65±0.05        | 0.70±0.03      | 0.63±0.04 | 0.70±0.04 | **0.53±0.05**  | 0.59±0.05    |
| 100% / 70%       | MAGIC     | **0.40±0.03**    | **0.46±0.03**  | **0.67±0.05**    | **0.76±0.03**  | 0.70±0.05 | **0.81±0.04** | **0.54±0.06**  | **0.66±0.05**|
| **70% / 100%**   |  ARF       | 0.07±0.02        | 0.08±0.02      | 0.25±0.05        | 0.29±0.05      | 0.24±0.02 | 0.33±0.01 | 0.26±0.05      | 0.29±0.04    |
| 70% / 100%       | ARF$_T$     | 0.21±0.05        | 0.26±0.04      | 0.22±0.06        | 0.28±0.06      | **0.73±0.00** | 0.73±0.00 | 0.41±0.06      | 0.44±0.06    |
| 70% / 100%       | cGRU      | 0.33±0.05        | 0.39±0.05      | 0.59±0.07        | 0.70±0.04      | 0.69±0.05 | **0.81±0.03** | 0.50±0.06      | 0.61±0.06    |
| 70% / 100%       | cPNN      | 0.34±0.02        | 0.34±0.03      | 0.65±0.03        | 0.70±0.02      | 0.59±0.07 | 0.66±0.06 | 0.50±0.05      | 0.53±0.05    |
| 70% / 100%       | MAGIC     | **0.43±0.03**    | **0.48±0.02**  | **0.67±0.04**    | **0.77±0.02**  | 0.69±0.04 | **0.81±0.03** | **0.55±0.05**  | **0.66±0.03**|

### Prequential Evaluation (Balanced Accuracy)

| **Precision / Recall** | Model | AirQuality Start | AirQuality End | PowerConsumption Start | PowerConsumption End | SRW Start   | SRW End     | Weather Start | Weather End   |
|------------------------|-------|------------------|----------------|-------------------------|----------------------|-------------|-------------|----------------|----------------|
| **100% / 100%**        | ARF   | 0.53±0.01        | 0.54±0.01      | 0.62±0.03               | 0.64±0.02            | 0.62±0.01   | 0.67±0.00   | 0.63±0.02      | 0.64±0.02      |
| 100% / 100             | ARF$_T$ | 0.61±0.02        | 0.63±0.02      | 0.61±0.03               | 0.64±0.03            | **0.87±0.00** | 0.87±0.00   | 0.71±0.03      | 0.72±0.03      |
| 100% / 100             | cGRU  | 0.66±0.03        | 0.69±0.03      | 0.79±0.04               | 0.85±0.02            | 0.84±0.03   | **0.90±0.02** | 0.75±0.03      | 0.80±0.03      |
| 100% / 100             | cPNN  | 0.67±0.01        | 0.67±0.01      | 0.83±0.01               | 0.85±0.01            | 0.80±0.03   | 0.84±0.02   | 0.76±0.03      | 0.78±0.03      |
| 100% / 100             | MAGIC | **0.71±0.02**     | **0.73±0.01**  | **0.84±0.02**           | **0.88±0.01**        | 0.86±0.02   | **0.91±0.02** | **0.78±0.02**  | **0.83±0.02**  |
| **100% / 70%**         | ARF   | 0.53±0.01        | 0.54±0.01      | 0.62±0.03               | 0.64±0.03            | 0.62±0.01   | 0.67±0.00   | 0.63±0.03      | 0.64±0.02      |
| 100% / 70%             | ARF$_T$ | 0.60±0.03        | 0.63±0.02      | 0.60±0.03               | 0.64±0.03            | **0.87±0.00** | 0.87±0.00   | 0.71±0.03      | 0.72±0.03      |
| 100% / 70%             | cGRU  | 0.66±0.03        | 0.69±0.03      | 0.79±0.04               | 0.85±0.02            | 0.84±0.03   | **0.90±0.02** | 0.75±0.03      | 0.80±0.03      |
| 100% / 70%             |cPNN  | 0.67±0.01        | 0.67±0.02      | **0.83±0.02**           | 0.85±0.02            | 0.81±0.02   | 0.85±0.02   | **0.77±0.03**  | 0.79±0.03      |
| 100% / 70%             | MAGIC | **0.70±0.02**     | **0.73±0.01**  | **0.83±0.02**           | **0.88±0.01**        | 0.85±0.02   | **0.91±0.02** | **0.77±0.03**  | **0.83±0.02**  |
| **70% / 100%**         |  ARF   | 0.53±0.01        | 0.54±0.01      | 0.62±0.03               | 0.64±0.02            | 0.62±0.01   | 0.67±0.00   | 0.63±0.02      | 0.64±0.02      |
| 70% / 100%             | ARF$_T$ | 0.61±0.02        | 0.63±0.02      | 0.61±0.03               | 0.64±0.03            | **0.87±0.00** | 0.87±0.00   | 0.71±0.03      | 0.72±0.03      |
| 70% / 100%             | cGRU  | 0.66±0.03        | 0.69±0.03      | 0.79±0.04               | 0.85±0.02            | 0.84±0.03   | **0.90±0.02** | 0.75±0.03      | 0.80±0.03      |
| 70% / 100%             | cPNN  | 0.67±0.01        | 0.67±0.01      | 0.82±0.01               | 0.85±0.01            | 0.79±0.03   | 0.83±0.03   | 0.75±0.02      | 0.76±0.03      |
| 70% / 100%             | MAGIC | **0.71±0.01**     | **0.74±0.01**  | **0.84±0.02**           | **0.89±0.01**        | 0.85±0.02   | **0.91±0.02** | **0.77±0.02**  | **0.83±0.02**  |

### CL Evaluation (Cohen's Kappa)

| **Precision / Recall**   | Model   | AirQuality AVG | AirQuality BWT | PowerConsumption AVG | PowerConsumption BWT | SRW AVG   | SRW BWT     | Weather AVG | Weather BWT |
|--------------------------|---------|----------------|----------------|-----------------------|----------------------|-----------|-------------|-------------|-------------|
| **100% prec / 100% rec** | ARF     | 0.01±0.01      | -0.07±0.02     | 0.07±0.02             | -0.35±0.08           | 0.09±0.02 | -0.36±0.03   | 0.04±0.01   | -0.18±0.04   |
| 100% prec / 100% rec     | ARF$_T$   | 0.14±0.04      | -0.19±0.05     | 0.09±0.02             | -0.31±0.12           | **0.73±0.00** | -0.00±0.00   | 0.20±0.05   | -0.24±0.06   |
| 100% prec / 100% rec     | cGRU    | 0.06±0.02      | -0.48±0.06     | 0.18±0.05             | -0.86±0.08           | 0.20±0.05 | -0.85±0.07   | 0.11±0.02   | -0.68±0.06   |
| 100% prec / 100% rec     | cPNN    | 0.31±0.05      | -0.09±0.05     | 0.51±0.09             | -0.32±0.15           | 0.59±0.10 | -0.21±0.12   | 0.42±0.09   | -0.19±0.09   |
| 100% prec / 100% rec     | MAGIC   | **0.41±0.06**  | -0.11±0.05     | **0.55±0.10**         | -0.35±0.15           | 0.64±0.11 | -0.26±0.14   | **0.49±0.07** | -0.24±0.08   |
| **100% prec / 70% rec**  | ARF     | 0.01±0.01      | -0.07±0.02     | 0.07±0.02             | -0.35±0.08           | 0.09±0.02 | -0.36±0.03   | 0.04±0.01   | -0.18±0.04   |
| 100% prec / 70% rec      | ARF$_T$   | 0.14±0.04      | -0.19±0.05     | 0.09±0.02             | -0.31±0.12           | **0.73±0.00** | -0.00±0.00   | 0.20±0.05   | -0.24±0.06   |
| 100% prec / 70% rec      | cGRU    | 0.06±0.02      | -0.48±0.06     | 0.18±0.05             | -0.86±0.08           | 0.20±0.05 | -0.85±0.07   | 0.11±0.02   | -0.68±0.06   |
| 100% prec / 70% rec      | cPNN    | 0.28±0.06      | -0.13±0.07     | 0.47±0.11             | -0.39±0.16           | 0.55±0.09 | -0.29±0.13   | 0.36±0.07   | -0.29±0.10   |
| 100% prec / 70% rec      | MAGIC   | **0.36±0.06**  | -0.16±0.07     | **0.50±0.09**         | -0.42±0.13           | 0.59±0.09 | -0.33±0.13   | **0.42±0.07** | -0.30±0.09   |
| **70% prec / 100% rec**  | ARF     | 0.01±0.01      | -0.07±0.02     | 0.07±0.02             | -0.35±0.08           | 0.09±0.02 | -0.36±0.03   | 0.04±0.01   | -0.18±0.04   |
| 70% prec / 100% rec      | ARF$_T$   | 0.14±0.04      | -0.19±0.05     | 0.09±0.02             | -0.31±0.12           | **0.73±0.00** | -0.00±0.00   | 0.20±0.05   | -0.24±0.06   |
| 100% prec / 100% rec     | cGRU    | 0.06±0.02      | -0.48±0.06     | 0.18±0.05             | -0.86±0.08           | 0.20±0.05 | -0.85±0.07   | 0.11±0.02   | -0.68±0.06   |
| 100% prec / 100% rec     | cPNN    | 0.34±0.05      | -0.05±0.05     | **0.60±0.08**         | -0.18±0.13           | 0.65±0.07 | -0.10±0.08   | 0.46±0.07   | -0.10±0.07   |
| 100% prec / 100% rec     | MAGIC   | **0.44±0.06**  | -0.10±0.07     | **0.61±0.09**         | -0.28±0.13           | **0.72±0.07** | -0.16±0.10   | **0.53±0.06** | -0.19±0.07   |

**BibTeX:**
```bibtex
@inproceedings{giannini_magic_net,
  author       = {Federico Giannini and
                  Sandro D'andrea and
                  Emanuele Della Valle},
  title        = {Don't Look Back in Anger: {MAGIC} Net for Streaming Continual Learning
                  with Temporal Dependence},
  booktitle    = {{IEEE} Big Data},
  pages        = {1396--1403},
  publisher    = {{IEEE}},
  year         = {2025}
}