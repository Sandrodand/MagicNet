import os
import pickle
from functools import partial
from river import forest, stream
from river import tree
from river.drift import ADWIN

from detectors.sentinel import Sentinel
from models.crnn.cgru import cGRULinear
from models.crnn.clstm import cLSTMLinear
from models.cpnn.cpnn import cPNN
from models.cpnn.dynamic_cpnn import DynamicCPNN
from evaluation.learner_config import LearnerConfig
from models.magic.magic_net import MagicNet
from models.mh_crnn.mhcrnn import MultiHeadCrnn
from models.sml.temporally_augmented_classifier import TemporallyAugmentedClassifier
from models.sml.temporally_augmented_features_classifier import (
    TemporallyAugmentedFeaturesClassifier,
)


_UNSET = object()


class Config:
    def __init__(self, base_learner="clstm", *, base_rnn=None):
        selected = base_learner if base_rnn is None else base_rnn
        if isinstance(selected, str):
            selected = [selected]
        self.base_rnn = [self._normalize_rnn(rnn) for rnn in selected]
        if not self.base_rnn or len(set(self.base_rnn)) != len(self.base_rnn):
            raise ValueError("Select at least one RNN without duplicates")
        self.ta_order = 9
        self.seq_len = 10
        self.num_features = 2
        self.batch_size = 128
        self.iterations = 1
        self.initial_task = 1
        self.eval_cl = None
        self.eval_preq = None
        self.path = None
        self.converters = None
        self.delta = 0.002
        self.output_size = 2
        self.hidden_size = {rnn: 50 for rnn in self.base_rnn}
        self.threshold_fn = "sigmoid"
        self.hidden_mult = {rnn: None for rnn in self.base_rnn}
        self.device = "cpu"
        self.mask_init = "uniform"
        self.ensemble_batches = {rnn: 30 for rnn in self.base_rnn}
        self.ensemble_th = {rnn: 1.05 for rnn in self.base_rnn}
        self.drift_delay = 1200
        self.save_column_freq = 500
        self.base_learner = {
            rnn: BaseLearner(partial(self.create_cpnn_for_dynamic, rnn=rnn))
            for rnn in self.base_rnn
        }

    @staticmethod
    def _normalize_rnn(rnn):
        aliases = {"gru": "gru", "cgru": "gru", "lstm": "lstm", "clstm": "lstm"}
        if not isinstance(rnn, str) or rnn.lower() not in aliases:
            raise ValueError("RNN must be 'gru' or 'lstm'")
        return aliases[rnn.lower()]

    def _resolve_rnn(self, rnn=None):
        if rnn is None:
            if len(self.base_rnn) != 1:
                raise ValueError("Select rnn explicitly when multiple RNNs are configured")
            return self.base_rnn[0]
        rnn = self._normalize_rnn(rnn)
        if rnn not in self.base_rnn:
            raise ValueError(f"RNN '{rnn}' is not enabled")
        return rnn

    def _update_rnn_parameter(self, name, value):
        if isinstance(value, dict):
            values = {self._normalize_rnn(key): item for key, item in value.items()}
            missing = set(self.base_rnn) - values.keys()
            if missing:
                raise ValueError(f"Missing {name} for: {', '.join(sorted(missing))}")
        else:
            values = {rnn: value for rnn in self.base_rnn}
        for rnn in self.base_rnn:
            if values[rnn] is not None or name == "hidden_mult":
                getattr(self, name)[rnn] = values[rnn]

    def set_params(
        self,
        ta_order=None,
        device=None,
        seq_len=None,
        num_features=None,
        batch_size=None,
        iterations=None,
        initial_task=None,
        path=None,
        converters=None,
        delta=None,
        output_size=None,
        hidden_size=None,
        threshold_fn=None,
        hidden_mult=None,
        ensemble_batches=None,
        ensemble_th=None,
        save_column_freq=_UNSET,
        drift_delay=_UNSET,
    ):
        previous = (self.path, self.num_features, self.output_size, self.seq_len,
                    self.batch_size, self.device)
        previous_hidden = self.hidden_size.copy()
        if ta_order is not None:
            self.ta_order = ta_order
        if device is not None:
            self.device = device
        if seq_len is not None:
            self.seq_len = seq_len
        if num_features is not None:
            self.num_features = num_features
        if batch_size is not None:
            self.batch_size = batch_size
        if iterations is not None:
            self.iterations = iterations
        if initial_task is not None:
            self.initial_task = initial_task
        if path is not None:
            self.path = path
        if converters is not None:
            self.converters = converters
        if delta is not None:
            self.delta = delta
        if output_size is not None:
            self.output_size = output_size
        if hidden_size is not None:
            self._update_rnn_parameter("hidden_size", hidden_size)
        if threshold_fn is not None:
            self.threshold_fn = threshold_fn
        if hidden_mult is not None:
            self._update_rnn_parameter("hidden_mult", hidden_mult)
        if ensemble_batches is not None:
            self._update_rnn_parameter("ensemble_batches", ensemble_batches)
        if ensemble_th is not None:
            self._update_rnn_parameter("ensemble_th", ensemble_th)
        if save_column_freq is not _UNSET:
            self.save_column_freq = save_column_freq
        if drift_delay is not _UNSET:
            self.drift_delay = drift_delay
        current = (self.path, self.num_features, self.output_size, self.seq_len,
                   self.batch_size, self.device)
        for rnn in self.base_rnn:
            if previous != current or previous_hidden[rnn] != self.hidden_size[rnn]:
                self.base_learner[rnn].invalidate()
            elif self.base_learner[rnn].base_learner is not None:
                self.base_learner[rnn].base_learner.set_save_column_freq(
                    self.save_column_freq, self.drift_delay
                )

    def initialize_callback(self, eval_cl_, eval_preq_):
        self.eval_cl = eval_cl_
        self.eval_preq = eval_preq_

    @staticmethod
    def create_hat():
        return tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=100,
            delta=1e-5,
            leaf_prediction="nb",
            nb_threshold=10,
        )

    def create_hat_ta(self):
        return TemporallyAugmentedClassifier(
            base_learner=self.create_hat(),
            num_old_labels=self.ta_order,
        )

    @staticmethod
    def create_arf():
        return forest.ARFClassifier(leaf_prediction="nb")

    @staticmethod
    def create_arf_no_adwin():
        return forest.ARFClassifier(
            leaf_prediction="nb", drift_detector=None, warning_detector=None
        )

    def create_arf_ta(self):
        return TemporallyAugmentedClassifier(
            base_learner=self.create_arf(),
            num_old_labels=self.ta_order,
        )

    def create_arf_ta_no_adwin(self):
        return TemporallyAugmentedClassifier(
            base_learner=self.create_arf_no_adwin(),
            num_old_labels=self.ta_order,
        )

    def _create_cpnn(self, rnn, quantize=False, checkpoints=True):
        rnn = self._resolve_rnn(rnn)
        column = cGRULinear if rnn == "gru" else cLSTMLinear
        return cPNN(
            column_class=column,
            device=self.device,
            seq_len=self.seq_len,
            acpnn=True,
            qcpnn=quantize,
            batch_size=self.batch_size,
            save_column_freq=self.save_column_freq if checkpoints else None,
            drift_delay=self.drift_delay,
            input_size=self.num_features,
            output_size=self.output_size,
            hidden_size=self.hidden_size[rnn],
        )

    def create_qcpnn_clstm(self):
        return self._create_cpnn("lstm", quantize=True, checkpoints=False)

    def create_acpnn_clstm(self):
        return self._create_cpnn("lstm", checkpoints=False)

    def create_cpnn_for_dynamic(self, rnn=None):
        return self._create_cpnn(rnn)

    def create_cpnn_clstm_for_dynamic(self):
        return self.create_cpnn_for_dynamic("lstm")

    def create_cpnn_cgru_for_dynamic(self):
        return self.create_cpnn_for_dynamic("gru")

    def create_base_cpnn(self, rnn=None):
        return self.base_learner[self._resolve_rnn(rnn)].get_cpnn()

    def create_dynamic_cpnn(self, rnn=None):
        rnn = self._resolve_rnn(rnn)
        return DynamicCPNN(
            [self.base_learner[rnn].get_base_learner()],
            drift_window_batches=self.ensemble_batches[rnn],
        )

    def _magic_parameters(self, rnn):
        rnn = self._resolve_rnn(rnn)
        return dict(
            device=self.device,
            seq_len=self.seq_len,
            input_size=self.num_features,
            train_verbose=False,
            batch_size=self.batch_size,
            hidden_size=self.hidden_size[rnn],
            hidden_mult=self.hidden_mult[rnn],
            ensemble_batches=self.ensemble_batches[rnn],
            ensemble_th=self.ensemble_th[rnn],
            base_learner_weights=self.base_learner[rnn].get_base_learner(),
            checkpoint_freq=self.save_column_freq,
            drift_delay=self.drift_delay,
            output_size=self.output_size,
        )

    def create_magic_net(self, rnn=None):
        return MagicNet(threshold_fn="sigmoid", multi_head=True, **self._magic_parameters(rnn))

    def create_multihead_crnn(self, rnn=None):
        return MultiHeadCrnn(**self._magic_parameters(rnn))

    def create_magic_net_binary(self, rnn=None):
        return MagicNet(threshold_fn="binarizer", multi_head=False, **self._magic_parameters(rnn))

    def create_magic_net_binary_mh(self, rnn=None):
        return MagicNet(threshold_fn="binarizer", multi_head=True, **self._magic_parameters(rnn))

    def create_magic_net_single_head(self, rnn=None):
        return MagicNet(threshold_fn="sigmoid", multi_head=False, **self._magic_parameters(rnn))

    def create_learners(self, rnn_methods, stream_methods=()):
        """Bind each factory to its architecture; non-RNN methods are created once."""
        rnn_factories = {
            "cRNN": (self.create_base_cpnn, dict(cpnn=True, drift=False)),
            "cPNN": (self.create_base_cpnn, dict(cpnn=True, drift=True)),
            "MAGIC_Net": (self.create_magic_net, dict(magic=True, drift=True)),
            "MAGIC_Net_sh": (self.create_magic_net_single_head, dict(magic=True, drift=True)),
            "MAGIC_Net_binary_sh": (self.create_magic_net_binary, dict(magic=True, drift=True)),
            "MAGIC_Net_binary_mh": (self.create_magic_net_binary_mh, dict(magic=True, drift=True)),
            "MultiHeadCRNN": (self.create_multihead_crnn, dict(magic=True, drift=True)),
            "DYNcPNN": (self.create_dynamic_cpnn, dict(dyn_cpnn=True)),
        }
        stream_factories = {
            "ARF": (self.create_arf, False),
            "ARF_TA": (self.create_arf_ta, True),
            "HAT": (self.create_hat, False),
            "HAT_TA": (self.create_hat_ta, True),
        }
        if len(set(rnn_methods)) != len(rnn_methods) or len(set(stream_methods)) != len(stream_methods):
            raise ValueError("Learner names must not be duplicated")
        if set(rnn_methods) - rnn_factories.keys() or set(stream_methods) - stream_factories.keys():
            raise ValueError("Unknown learner name")
        learners = [
            LearnerConfig(name, model=stream_factories[name][0], numeric=False, drift=False,
                          temp_dep=stream_factories[name][1])
            for name in stream_methods
        ]
        for rnn in self.base_rnn:
            for name in rnn_methods:
                factory, flags = rnn_factories[name]
                learners.append(LearnerConfig(f"{name}_{rnn}", model=partial(factory, rnn=rnn), **flags))
        if not learners:
            raise ValueError("Select at least one learner")
        return learners

    def _validate_base_learner(self, model, rnn):
        if not isinstance(model, cPNN):
            raise ValueError("Saved base learner must be a cPNN")
        column = model.columns.columns[0]
        expected = cGRULinear if rnn == "gru" else cLSTMLinear
        if not isinstance(column, expected):
            raise ValueError(f"Saved base learner is not a {rnn.upper()}")
        dimensions = (column.input_size, column.hidden_size, column.output_size,
                      model.seq_len, model.batch_size)
        wanted = (self.num_features, self.hidden_size[rnn], self.output_size,
                  self.seq_len, self.batch_size)
        if dimensions != wanted or len(model.columns.columns) != 1:
            raise ValueError(f"Saved {rnn.upper()} base learner does not match the configured dimensions")

    def prepare_base_learners(self, root, dataset, load=True, save=True):
        """Load or initialize each architecture independently after dataset setup."""
        for rnn in self.base_rnn:
            filename = os.path.join(root, rnn, f"{dataset}.pkl")
            model = None
            if load:
                try:
                    with open(filename, "rb") as stream:
                        model = pickle.load(stream)
                except FileNotFoundError:
                    # Flat legacy files are unambiguous only after checking their RNN type.
                    legacy = os.path.join(root, f"{dataset}.pkl")
                    if os.path.isfile(legacy):
                        with open(legacy, "rb") as stream:
                            candidate = pickle.load(stream)
                        expected = cGRULinear if rnn == "gru" else cLSTMLinear
                        if isinstance(candidate, cPNN) and isinstance(candidate.columns.columns[0], expected):
                            model = candidate
            if model is not None:
                self._validate_base_learner(model, rnn)
                model.set_save_column_freq(self.save_column_freq, self.drift_delay)
                self.base_learner[rnn].set_base_learner(model)
                print(f"{rnn.upper()} base learner loaded.")
            else:
                self.base_learner[rnn].reset_base_learner()
                print(f"{rnn.upper()} base learner initialized.")
            if save and (not load or not os.path.isfile(filename)):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wb") as stream:
                    pickle.dump(self.base_learner[rnn].get_base_learner(), stream)

    def callback_func_cl(self, **kwargs):
        if "iteration" in kwargs:
            iteration = kwargs["iteration"]
        else:
            iteration = None
        if iteration is None:
            iterations = (0, self.iterations)
        else:
            iterations = (iteration, iteration + 1)
        self.eval_cl.evaluate(iterations)

    @staticmethod
    def callback_func_federated(**kwargs):
        if "suffix" in kwargs:
            iteration = kwargs["suffix"]
        else:
            iteration = None
        if iteration is None:
            iteration = 1
        selection = {}
        models = kwargs["models"]
        for model in models:
            if "F-cPNN" in model:
                selection[model] = {
                    "columns_task_ids": [
                        models[model].ensemble[i].task_ids
                        for i in range(len(models[model].ensemble))
                    ],
                    "federated_task_dict": models[model].task_dict,
                    "columns_perf": [
                        {
                            task: perf.get()
                            for task, perf in zip(m.task_ids, m.columns_perf)
                        }
                        for m in models[model].ensemble
                    ],
                }
        with open(os.path.join(kwargs["path"], f"task_ids{iteration}.pkl"), "wb") as f:
            pickle.dump(selection, f)

    def create_iter_csv(self):
        return stream.iter_csv(
            str(self.path) + ".csv", converters=self.converters, target="target"
        )

    def create_drift_detector(self):
        path = self.path.lower()
        if "future" in path:
            if "air_quality" in path:
                return Sentinel(
                    ADWIN(delta=self.delta, clock=1),
                    training_data_points=-1,
                    evaluator=self.create_arf_no_adwin(),
                )
            if "weather" in path:
                return Sentinel(
                    ADWIN(delta=self.delta, clock=1),
                    training_data_points=50 * 128,
                    evaluator=self.create_arf_no_adwin(),
                )
            if "energy" in path:
                return Sentinel(
                    ADWIN(delta=self.delta, clock=1),
                    training_data_points=50 * 128,
                    evaluator=self.create_arf_no_adwin(),
                )
        # not future
        if "air_quality" in path:
            return Sentinel(
                ADWIN(delta=self.delta, clock=1),
                training_data_points=-1,
                evaluator=self.create_arf_no_adwin(),
            )
        if "weather" in path:
            return Sentinel(
                ADWIN(delta=self.delta, clock=1),
                training_data_points=50 * 128,
                evaluator=self.create_arf_ta_features_no_adwin(),
            )
        if "energy" in path:
            return Sentinel(
                ADWIN(delta=self.delta, clock=1),
                training_data_points=50 * 128,
                evaluator=self.create_arf_ta_features_no_adwin(),
            )
        if "sine" in path:
            return Sentinel(
                ADWIN(delta=self.delta, clock=1),
                training_data_points=50 * 128,
                evaluator=self.create_arf_no_adwin(),
            )
        # else
        return Sentinel(
            ADWIN(delta=self.delta, clock=1),
            training_data_points=50 * 128,
            evaluator=self.create_arf_no_adwin(),
        )

    @staticmethod
    def create_hat():
        return tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=100,
            delta=1e-5,
            leaf_prediction="nb",
            nb_threshold=10,
        )

    def create_hat_ta(self):
        return TemporallyAugmentedClassifier(
            base_learner=self.create_hat(),
            num_old_labels=self.ta_order,
        )

    @staticmethod
    def create_arf():
        return forest.ARFClassifier(leaf_prediction="nb")

    @staticmethod
    def create_arf_no_adwin():
        return forest.ARFClassifier(
            leaf_prediction="nb", drift_detector=None, warning_detector=None
        )

    def create_arf_ta(self):
        return TemporallyAugmentedClassifier(
            base_learner=self.create_arf(),
            num_old_labels=self.ta_order,
        )

    def create_arf_ta_no_adwin(self):
        return TemporallyAugmentedClassifier(
            base_learner=self.create_arf_no_adwin(),
            num_old_labels=self.ta_order,
        )

    def create_arf_ta_features(self):
        return TemporallyAugmentedFeaturesClassifier(
            base_learner=self.create_arf(), ta_order=self.ta_order
        )

    def create_arf_ta_features_no_adwin(self):
        return TemporallyAugmentedFeaturesClassifier(
            base_learner=self.create_arf_no_adwin(), ta_order=self.ta_order
        )


class BaseLearner:
    def __init__(self, learner_func):
        self.learner_func = learner_func
        self.base_learner = None

    def get_base_learner(self):
        if self.base_learner is None:
            self.reset_base_learner()
        return pickle.loads(pickle.dumps(self.base_learner))

    def get_cpnn(self):
        return self.get_base_learner()

    def reset_base_learner(self):
        self.base_learner = self.learner_func()

    def invalidate(self):
        self.base_learner = None

    def set_base_learner(self, base_learner):
        self.base_learner = pickle.loads(pickle.dumps(base_learner))
