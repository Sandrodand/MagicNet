import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from models.magic.manager import *


class MagicNet:
    """Online recurrent classifier that adapts to drift with masks and expansion."""

    def __init__(
        self,
        model_class=None,
        device=None,
        lr: float = 0.01,
        seq_len: int = 5,
        input_size=4,
        train_epochs: int = 10,
        train_verbose: bool = False,
        batch_size: int = 128,
        reset_data_points: bool = False,
        threshold_fn="sigmoid",
        hidden_size=25,
        hidden_mult=None,
        ensemble_batches=50,
        ensemble_th=1.05,
        base_learner_weights=None,
        multi_head=True,
        ignore_option=True,
        expand_option=True,
        checkpoint_freq=500,
        drift_delay=1500,
        grace_period=5000,
        manager_class=MagicManager,
        data_points_previous_concept=1000,
        minimum_k_difference=0.04,
        name="MAGIC_Net",
        selection_batches=40,
        **kwargs,
    ):
        """
        Parameters
        ----------

        input_size: int, default: 4.
            Number of features per data point.
        model_class: class, default: None.
            Masked recurrent architecture. Inferred from the base learner, or GRU
            when no base learner is supplied. Accepts PiggyBackGRU/PiggyBackLSTM.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of the Adam Optimizer.
        seq_len: int, default: 5.
            The length of the sliding window that builds the single sequences.
            Must be greater than 1.
        train_epochs: int, default: 10.
            The training epochs to perform in _learn_many method.
        train_verbose: bool, default:False.
            True if, during the _learn_many execution, you want to print the metrics after each training epoch.
        batch_size: int, default: 128.
            The training mini-batch size. If calling learn_one method, the model will accumulate batch_size data point
            before performing the training.
        reset_data_points: bool, default: False.
            If True, after adding a drift, the buffer is cleared. This way the model cannot predict the labels
            associated with the first W-1 data points following the detected drift.
        threshold_fn: str, default: "sigmoid".
            The function to transform the real-weighted masks before applying them on the weights.
            Functions can be "binarizer", "ternarizer", "sigmoid", "tanh".
        hidden_size: int, default: 25.
            The starting hidden size of the model's cRNN.
        hidden_mult: int, default: None.
            Number of neurons added during expansion. None uses half the actual
            backbone hidden size, including when loaded from a shared base learner.
        ensemble_batches: int, default: 50.
            The number of batches used to train the model after a drift before choosing which option in the ensemble
            to keep.
        selection_batches: int, default: 40.
            Rolling minibatches used for online and final candidate selection.
        ensemble_th: float, default: 1.05.
            Expand must exceed both the best non-expansion Kappa and that score
            multiplied by this factor.
        base_learner_weights: cPNN, cGRULinear, or cLSTMLinear, optional.
            Reuse the recurrent weights and classifier of a pretrained learner.
        multi_head: bool, default: True.
            Use a separate linear head for each retained task.
        ignore_option: bool, default: True.
            Include the candidate that continues training without adapting the architecture.
        expand_option: bool, default: True.
            Include the candidate that increases the recurrent hidden size.
        checkpoint_freq: int, default: 500.
            Interval in labeled data points between rollback snapshots.
        drift_delay: int, default: 1500.
            Offset in data points used to look up a pre-drift snapshot.
        grace_period: int, default: 5000.
            Number of labeled points for which drift alarms are suppressed at startup
            and after accepting an adaptation.
        manager_class: class, default: MagicManager.
            Lifecycle manager. The head-only ablation supplies its own manager.
        data_points_previous_concept: int, default: 1000.
            Maximum number of labeled samples retained for the next forgetting check.
        minimum_k_difference: float, default: 0.04.
            Minimum Kappa advantage required to archive the pre-drift model after Ignore.
        name: str, default: "MAGIC_Net".
            Model identifier used in diagnostic messages.
        kwargs:
            Additional keyword arguments passed to the lifecycle manager, such as output_size.
        """
        if "initial_task_id" in kwargs:
            raise TypeError(
                "initial_task_id is no longer configurable; the first task is 1"
            )
        if "stride" in kwargs:
            raise TypeError("stride is no longer configurable; it is always 1")
        if seq_len < 2:
            raise ValueError("seq_len must be greater than 1")

        self.multihead = multi_head
        self.columns_args = kwargs
        if device is None:
            device = "cpu"
        self.device = device

        self.columns_args["device"] = device
        self.columns_args["lr"] = lr
        self.columns_args["batch_size"] = batch_size
        self.columns_args["train_epochs"] = train_epochs
        self.columns_args["train_verbose"] = train_verbose
        self.columns_args["seq_len"] = seq_len
        self.columns_args["model_class"] = model_class
        self.ensemble_batches = ensemble_batches
        self.selection_batches = selection_batches
        self.ensemble_th = ensemble_th

        self.seq_len = seq_len
        self.stride = 1
        self.task_ids = [1]
        self.previous_data_points_anytime_inference = None
        self.previous_data_points_anytime_hidden = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_batch_train = None
        self.previous_data_points_batch_inference = None
        self.x_batch = []
        self.y_batch = []
        self.batch_size = batch_size
        self.saved_columns = []
        self.data_point_counter = 1
        self.reset_data_points = reset_data_points
        self.grace_period = grace_period
        self.ignore_option = ignore_option
        self.expand_option = expand_option

        self.checkpoint_freq = checkpoint_freq

        self.manager = manager_class(
            name=name,
            ensemble_batches=ensemble_batches,
            selection_batches=selection_batches,
            input_size=input_size,
            threshold_fn=threshold_fn,
            hidden_size=hidden_size,
            hidden_mult=hidden_mult,
            ensemble_th=ensemble_th,
            base_learner_weights=base_learner_weights,
            multi_head=self.multihead,
            ignore_option=ignore_option,
            expand_option=expand_option,
            checkpoint_freq=checkpoint_freq,
            drift_delay=drift_delay,
            grace_period=grace_period,
            data_points_previous_concept=data_points_previous_concept,
            minimum_k_difference=minimum_k_difference,
            **self.columns_args,
        )

    @property
    def name(self):
        """Return the model name used in diagnostic output."""
        return self.manager.name

    @name.setter
    def name(self, value):
        """Set the model name used in diagnostic output."""
        self.manager.name = value

    def get_seq_len(self):
        """Return the number of observations in a recurrent input window."""
        return self.seq_len

    def _cut_in_sequences(self, x, y):
        seqs_features = []
        seqs_targets = []
        for i in range(0, len(x), self.stride):
            if len(x) - i >= self.seq_len:
                seqs_features.append(x[i : i + self.seq_len, :].astype(np.float32))
                if y is not None:
                    seqs_targets.append(
                        np.asarray(y[i : i + self.seq_len], dtype=np.int_)
                    )
        return np.asarray(seqs_features), np.asarray(seqs_targets)

    def _convert_to_tensor_dataset(self, x, y=None):
        """
        It converts the dataset in order to be inputted to MAGIC Net, by building the different sequences and
        converting them to TensorDataset.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: list, default: None
            The target values of the batch. If None only features will be loaded.
        Returns
        -------
        dataset: torch.data_utils.TensorDataset
            The tensor dataset representing the different sequences.
            The features values have shape: (batch_size - seq_len + 1, seq_len, n_features)
            The target values have shape: (batch_size - seq_len + 1, seq_len)
        """
        x, y = self._cut_in_sequences(x, y)
        x = torch.tensor(x)
        if len(y) > 0:
            y = torch.tensor(y).type(torch.LongTensor)
            return data_utils.TensorDataset(x, y)
        return x

    def _load_batch(self, x: np.array, y: np.array = None):
        """
        It transforms the batch in order to be inputted to Magic Net, by building the different sequences and
        converting them to tensors.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: list, default: None.
            The target values of the batch. If None only features will be loaded.
        Returns
        -------
        x: torch.Tensor
            The features values of the created sequences. It has shape: (batch_size - seq_len + 1, seq_len, n_features)
        y: torch.Tensor
            The target values of the samples in the batc. It has length: batch_size. If y is None it returns None.
        y_seq: torch.Tensor
            The target values of the created sequences. It has shape: (batch_size - seq_len + 1, seq_len). If y is None it returns None.
        """
        batch = self._convert_to_tensor_dataset(x, y)
        batch_loader = DataLoader(
            batch, batch_size=batch.tensors[0].size()[0], drop_last=False, shuffle=False
        )
        y_seq = None
        for x, y_seq in batch_loader:  # only to take x and y from loader
            break
        y = torch.tensor(y)
        return x, y, y_seq

    def learn_one(
        self,
        x: np.array,
        y: int,
        previous_data_points: np.array = None,
        timestamp: int = -1,
    ):
        """Buffer one labeled observation and train when the minibatch is full.

        During a drift comparison, update candidate Kappa scores using the
        prediction stored for timestamp. Use the same timestamp in predict_one
        and learn_one for delayed labels. The default -1 is for immediate
        prediction and label pairs. previous_data_points is a legacy argument.
        Returns None.
        """

        self.manager.collect_concept_sample(x, y)
        self.x_batch.append(x)
        self.y_batch.append(y)
        self.data_point_counter += 1
        was_in_expansion = self.manager.in_expansion
        if self.manager.in_expansion is True:
            self.manager.update_metrics(y, timestamp=timestamp)
        if len(self.x_batch) == self.batch_size:
            self._learn_many(np.array(self.x_batch), np.array(self.y_batch))
            self.x_batch = []
            self.y_batch = []
        self.manager.update_counter(
            self.data_point_counter, advance_grace_period=not was_in_expansion
        )
        return None

    def add_new_column(self, task_id=None):
        """Handle a drift notification and start a comparison when eligible.

        An active comparison or grace period suppresses the notification.
        task_id defaults to the previous ID plus one. Supply consecutive IDs
        when assigning them explicitly. Starting a comparison clears the
        pending training minibatch. Temporal context is cleared when
        reset_data_points is enabled.
        """
        if self.manager.in_expansion or self.manager.in_grace_period:
            self.manager.add_new_column(None)
            return

        if task_id is None:
            task_id = self.task_ids[-1] + 1

        self.task_ids.append(task_id)
        self.manager.add_new_column(task_id)

        if self.reset_data_points:
            self.reset_previous_data_points()

        self.x_batch = []
        self.y_batch = []

    def _learn_many(self, x: np.array, y: np.array) -> dict:
        """
        It trains Magic Net on a single mini-batch of data points.


        Parameters
        ----------
        x: numpy.array or list
            The features values of the mini-batch.
        y: np.array or list
            The target values of the mini-batch.

        Returns
        -------
        perf_train: dict
            The dictionary representing training's performance. Each key contains the list representing all the epochs' performances.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
            For each metric the dict contains a list of epochs' values.
        """

        x = np.array(x)
        y = list(y)
        if x.shape[0] < self.get_seq_len():
            return {}

        if self.previous_data_points_batch_train is not None:
            x = np.concatenate([self.previous_data_points_batch_train, x], axis=0)
            y = np.concatenate([[i for i in range(self.seq_len - 1)], y], axis=0)
        self.previous_data_points_batch_train = x[-(self.seq_len - 1) :]
        x, y, _ = self._load_batch(x, y)
        y = y[self.seq_len - 1 :]

        perf_train = self.manager.train(x, y)

        return perf_train

    def reset_previous_data_points(self):
        """Clear the temporal context used to assemble recurrent sequences."""
        self.previous_data_points_batch_train = None
        self.previous_data_points_batch_inference = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_anytime_inference = None
        self.previous_data_points_anytime_hidden = None

    def _single_data_point_prep(
        self, x, previous_data_points_param: np.array = None, inference=True
    ):
        x = np.array(x).reshape(1, -1)
        if inference:
            previous_data_points = self.previous_data_points_anytime_inference
        else:
            previous_data_points = self.previous_data_points_anytime_hidden
        if previous_data_points_param is not None:
            previous_data_points = previous_data_points_param
        if previous_data_points is None:
            previous_data_points = x
            if inference:
                self.previous_data_points_anytime_inference = previous_data_points
            else:
                self.previous_data_points_anytime_hidden = previous_data_points
            return None
        if len(previous_data_points) != self.seq_len - 1:
            previous_data_points = np.concatenate([previous_data_points, x])
            if inference:
                self.previous_data_points_anytime_inference = previous_data_points
            else:
                self.previous_data_points_anytime_hidden = previous_data_points
            return None
        previous_data_points = np.concatenate([previous_data_points, x])
        x = self._convert_to_tensor_dataset(previous_data_points).to(self.device)
        previous_data_points = previous_data_points[1:]
        if inference:
            self.previous_data_points_anytime_inference = previous_data_points
        else:
            self.previous_data_points_anytime_hidden = previous_data_points
        return x

    def predict_one(
        self,
        x: np.array,
        previous_data_points: np.array = None,
        task_id=None,
        timestamp: int = -1,
    ):
        """Predict the label of x, or return None during sequence warm-up.

        previous_data_points supplies the preceding feature rows. When it is
        omitted, use the accumulated prediction context. task_id selects a
        head and defaults to the current task. During a comparison, return
        the prediction of the candidate with the highest selection-window
        Kappa and store all candidate labels under timestamp for learn_one.
        """
        x = self._single_data_point_prep(x, previous_data_points)
        if x is None:
            # No prediction exists for this timestamp during sequence warm-up.
            self.manager.pending_predictions.pop(timestamp, None)
            return None
        with torch.no_grad():
            pred, _ = get_pred_from_outputs(
                self.manager.predict_one(x, task_id=task_id, timestamp=timestamp)
            )
            pred = int(pred[-1].detach().cpu().numpy())
        return pred

    def get_state_dict(self):
        """Return the active model's weights, or None during a drift comparison."""
        return self.manager.get_state_dict()

    def get_size(self):
        """Return serialized weight size in MB for the active model or ensemble."""
        if self.manager.in_expansion:
            size = 0
            for k in self.manager.ensemble:
                size += compute_model_size(self.manager.ensemble[k])
            return size
        return compute_model_size(self.manager.model)

    def get_seq_len(self):
        """Return the number of observations in a recurrent input window."""
        return self.seq_len
