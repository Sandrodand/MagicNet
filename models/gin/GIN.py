import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from models.gin.manager import *


class GIN:
    """
    Class that implements all the cPNN structure.
    """

    def __init__(
        self,
        model_class=PiggyBackGRU,
        device=None,
        lr: float = 0.01,
        seq_len: int = 5,
        stride: int = 1,
        base_model="gru",
        mask_init="1s",
        input_size=4,
        train_epochs: int = 10,
        train_verbose: bool = False,
        initial_task_id: int = 1,
        batch_size: int = 128,
        save_column_freq: int = None,
        reset_data_points: bool = False,
        threshold_fn="binarizer",
        hidden_size=50,
        hidden_mult=50,
        ensemble_batches=50,
        ensemble_th=1.1,
        ensemble_mode="classic",
        cGRU_weights=None,
        cap_sigmoid=True,
        expand_last=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        column_class: default: !!!!.
            The class that implements the column.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of single columns' Adam Optimizer.
        seq_len: int, default: 5.
            The length of the sliding window that builds the single sequences.
        stride: int, default: 1.
            The length of the sliding window's stride.
        train_epochs: int, default: 10.
            In case of anytime_learner=False, the training epochs to perform in learn_many method.
        train_verbose: bool, default:False.
            True if, during the learn_many execution, you want to print the metrics after each training epoch.
        anytime_learner: bool, default: False.
            If True the model learns data point by data point by data point. Otherwise, it buffers mini-batches of
            data points to perform training.
        acpnn: bool, default: False.
            In case of anytime_learner = False is always True.
            If True, the model is an Anytime cPNN. It considers only past temporal dependencies. The model
            is a many_to_one model and each data point's prediction is associated to the first sequence in which
            it appears. If False, the model considers both past and future temporal dependencies.
            The model is a many_to_many model and each data point's prediction is the average prediction between all the
            sequences in which it appears.
        qcpnn: bool, default: False
            If True the model is a QcPNN. After a concept drift, the last column is quantized.
        initial_task_id: int, default: 1.
            The id of the first task. It can be ignored.
        batch_size: int, default: 128.
            The training mini-batch size. If calling learn_one method, the model will accumulate batch_size data point
            before performing the training.
        save_column_freq: int, default: None
            The frequency at which the model stores the current state of the last column. Since the drift detector
            may have a delay, before adding a new column, the stored state of the last column is restored. This avoids
            to use a column that does not represent the concept.
        reset_data_points: bool, default: False.
            In case of acpnn, when predicting the label of the data point at timestamp t, the model buffers the data
            points from t-W to t-1 (where W is the sequence length). If True, after adding a new column, the buffer
            is cleared. This way the model cannot predict the labels associated with the first W-1 data points
            following the detected drift.
        kwargs:
            Parameters of column_class.
        """
        self.many_to_one = True
        self.cap_sigmoid = True

        self.columns_args = kwargs
        if device is None:
            device = "cpu"
        self.device = device

        self.columns_args["device"] = device
        self.columns_args["lr"] = lr
        self.columns_args["many_to_one"] = self.many_to_one
        self.columns_args["batch_size"] = batch_size
        self.columns_args["train_epochs"] = train_epochs
        self.columns_args["train_verbose"] = train_verbose
        self.columns_args["seq_len"] = seq_len
        self.columns_args["model_class"] = model_class
        self.columns_args["initial_task_id"] = initial_task_id

        # self.columns = cPNNColumns(**self.columns_args) # CHANGE

        self.base_model = base_model
        self.mask_init = mask_init
        self.ensemble_batches = ensemble_batches
        self.ensemble_th = ensemble_th
        self.ensemble_mode = ensemble_mode

        self.seq_len = seq_len
        self.stride = stride
        self.columns_perf = [metrics.CohenKappa()]
        self.task_ids = [initial_task_id]
        self.previous_data_points_anytime_inference = None
        self.previous_data_points_anytime_hidden = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_batch_train = None
        self.previous_data_points_batch_inference = None
        self.x_batch = []
        self.y_batch = []
        self.batch_size = batch_size
        self.save_column_freq = save_column_freq
        self.saved_columns = []
        self.cont = 1
        self.train_cont = [0]
        self.reset_data_points = reset_data_points
        self.predictions = {}
        """if model_class==PiggyBackGRU:
            self.model = PiggyBackGRU(hidden_size=self.hidden_size,base_model=base_model,seq_len=seq_len,mask_init=mask_init)
        """

        self.manager = CPGmanager(
            **self.columns_args,
            base_model=base_model,
            mask_init=mask_init,
            input_size=input_size,
            hidden_size=hidden_size,
            hidden_mult=hidden_mult,
            threshold_fn=threshold_fn,
            cGRU_weights=cGRU_weights,
            ensemble_batches=ensemble_batches,
            ensemble_th=ensemble_th,
            ensemble_mode=ensemble_mode,
            cap_sigmoid=cap_sigmoid,
            expand_last=expand_last
        )

    def get_seq_len(self):
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
        It converts the dataset in order to be inputted to GIN, by building the different sequences and
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
        It transforms the batch in order to be inputted to GIN, by building the different sequences and
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
        """
        It trains GIN on a single data point. The training is performed after filling
        up a mini_batch containing batch_size data points.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the single data point.
        y: int
            The target value of the single data point.
        previous_data_points: numpy.array, default: None.
            The features value of the data points preceding x in the sequence.
            If None, it uses the last seq_len-1 points seen during the last calls of the method.
            It returns None if the model has not seen yet seq_len-1 data points and previous_data_points is None.
        """

        self.x_batch.append(x)
        self.y_batch.append(y)
        self.cont += 1
        if self.manager.in_expansion is True:
            self.manager.update_metrics(y)
        if len(self.x_batch) == self.batch_size:
            self.learn_many(np.array(self.x_batch), np.array(self.y_batch))
            self.x_batch = []
            self.y_batch = []
        """if self.save_column_freq is not None:
            if self.cont % self.save_column_freq == 0:
                self.saved_columns.append(
                    {
                        "cont": self.cont,
                        "column": pickle.loads(
                            pickle.dumps(self.columns.columns[-1])
                        ),
                    }
                )
                self.saved_columns = self.saved_columns[-4:]"""
        return None

    def add_new_column(self, task_id=None):
        """
        !!! Signal concept drift and create the 2 possible models

        Parameters
        ----------
        task_id: int, default: None
            The id of the new task. If None it increments the last one.
        """
        if self.manager.in_expansion:
            self.manager.add_new_column(None)
            return

        if task_id is None:
            task_id = self.task_ids[-1] + 1

        self.task_ids.append(task_id)
        self.manager.add_new_column(task_id)

        if self.reset_data_points:
            self.reset_previous_data_points()
        self.columns_perf.append(metrics.CohenKappa())
        self.train_cont.append(0)
        self.x_batch = []
        self.y_batch = []

    def learn_many(self, x: np.array, y: np.array) -> dict:
        """
        It trains GIN on a single mini-batch of data points.


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

        self.train_cont[-1] = self.train_cont[-1] + 1
        return perf_train

    def reset_previous_data_points(self):
        self.previous_data_points_batch_train = None
        self.previous_data_points_batch_inference = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_anytime_inference = None
        self.previous_data_points_anytime_hidden = None
        self.predictions = {}

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

    def predict_one(self, x: np.array, previous_data_points: np.array = None):
        """
        It performs prediction on a single data point.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the single data point.
        previous_data_points: numpy.array, default: None.
            The features value of the data points preceding x in the sequence.
            If None, it uses the last seq_len-1 points seen during the last calls of the method.
            It returns None if the model has not seen yet seq_len-1 data points and previous_data_points is None.
        Returns
        -------
        prediction : int
            The predicted int label of x.
        """
        x = self._single_data_point_prep(x, previous_data_points)
        if x is None:
            return None
        with torch.no_grad():
            pred, _ = get_pred_from_outputs(self.manager.predict_one(x))
            pred = int(pred[-1].detach().cpu().numpy())
        return pred

    def get_state_dict(self):
        return self.manager.get_state_dict()

    def get_size(self):
        if self.manager.in_expansion:
            size = 0
            for k in self.manager.ensemble:
                size += compute_model_size(self.manager.ensemble[k])
            return size
        return compute_model_size(self.manager.model)
