import os
import pickle
from river import forest, stream
from river import tree
from river.drift import ADWIN

from detectors.sentinel import Sentinel
from models.crnn.cgru import cGRULinear
from models.crnn.clstm import cLSTMLinear
from models.cpnn.cpnn import cPNN
from models.magic.magic_net import MagicNet
from models.sml.temporally_augmented_classifier import TemporallyAugmentedClassifier
from models.sml.temporally_augmented_features_classifier import TemporallyAugmentedFeaturesClassifier


class Config:
    def __init__(self, base_learner="clstm"):
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
        self.hidden_size = 50
        self.threshold_fn = "ternarizer"
        self.hidden_mult = 1.5
        self.device = "cpu"
        self.mask_init = "uniform"
        self.ensemble_batches = 50
        self.ensemble_th = 1.2
        if "gru" in base_learner.lower():
            self.base_learner = BaseLearner(self.create_cpnn_cgru_for_dynamic)
        else:
            self.base_learner = BaseLearner(self.create_cpnn_clstm_for_dynamic)

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
        output_size=2,
        hidden_size=None,
        threshold_fn=None,
        mask_init=None,
        hidden_mult=None,
        ensemble_batches=None,
        ensemble_th=None,
    ):
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
            self.hidden_size = hidden_size
        if threshold_fn is not None:
            self.threshold_fn = threshold_fn
        if mask_init is not None:
            self.mask_init = mask_init
        if hidden_mult is not None:
            self.hidden_mult = hidden_mult
        if ensemble_batches is not None:
            self.ensemble_batches = ensemble_batches
        if ensemble_th is not None:
            self.ensemble_th = ensemble_th

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

    def create_qcpnn_clstm(self):
        return cPNN(
            column_class=cLSTMLinear,
            device="cpu",
            seq_len=self.seq_len,
            train_verbose=False,
            acpnn=True,
            qcpnn=True,
            batch_size=self.batch_size,
            save_column_freq=None,
            input_size=self.num_features,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )

    def create_acpnn_clstm(self):
        return cPNN(
            column_class=cLSTMLinear,
            device="cpu",
            seq_len=self.seq_len,
            train_verbose=False,
            acpnn=True,
            qcpnn=False,
            batch_size=self.batch_size,
            save_column_freq=None,
            input_size=self.num_features,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )

    def create_cpnn_clstm_for_dynamic(self):
        return cPNN(
            column_class=cLSTMLinear,
            device="cpu",
            seq_len=self.seq_len,
            train_verbose=False,
            acpnn=True,
            qcpnn=False,
            batch_size=self.batch_size,
            save_column_freq=2 * 10**3,
            input_size=self.num_features,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )

    def create_cpnn_cgru_for_dynamic(self):
        return cPNN(
            column_class=cGRULinear,
            device=self.device,
            seq_len=self.seq_len,
            train_verbose=False,
            acpnn=True,
            qcpnn=False,
            batch_size=self.batch_size,
            save_column_freq=1500,
            drift_delay=3000,
            input_size=self.num_features,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )

    def create_magic_net(self):
        return MagicNet(device="cpu", seq_len=self.seq_len, mask_init=self.mask_init, input_size=self.num_features,
                        train_verbose=False, initial_task_id=1, batch_size=self.batch_size,
                        threshold_fn=self.threshold_fn, hidden_size=self.hidden_size, hidden_mult=self.hidden_mult,
                        ensemble_batches=self.ensemble_batches, ensemble_th=self.ensemble_th,
                        cgru_weights=self.base_learner.get_base_learner(), cap_sigmoid=True, multi_head=True,
                        ignore_option=True, output_size=self.output_size, drift_delay=3000, checkpoint_freq=1500)

    def create_magic_net_no_rollback(self):
        return MagicNet(device="cpu", seq_len=self.seq_len, mask_init=self.mask_init, input_size=self.num_features,
                        train_verbose=False, initial_task_id=1, batch_size=self.batch_size,
                        threshold_fn=self.threshold_fn, hidden_size=self.hidden_size, hidden_mult=self.hidden_mult,
                        ensemble_batches=self.ensemble_batches, ensemble_th=self.ensemble_th,
                        cgru_weights=self.base_learner.get_base_learner(), cap_sigmoid=True, multi_head=True,
                        ignore_option=True, output_size=self.output_size, drift_delay=3000, checkpoint_freq=None)

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
        self.base_learner: cPNN = learner_func()

    def get_base_learner(self):
        return pickle.loads(pickle.dumps(self.base_learner))

    def get_cpnn(self):
        model: cPNN = pickle.loads(pickle.dumps(self.base_learner))
        #model.set_save_column_freq(save_column_freq=None)
        return model

    def reset_base_learner(self):
        self.base_learner = self.learner_func()

    def set_base_learner(self, base_learner):
        self.base_learner = base_learner
