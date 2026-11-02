from river import metrics
import pickle

from evaluation.evaluation_utils import compute_model_size
from models.magic.piggyback_cgru import (
    PiggyBackGRU,
)
from models.utils_scl.utils import *


class MagicManager(object):
    """
    Handles the lifecycle, training, masking, ensemble evaluation, and structural
    adaptation of the MAGIC Net architecture.
    """

    def __init__(
        self,
        model_class=PiggyBackGRU,
        device=None,
        ensemble_batches=30,
        lr=0.01,
        input_size=4,
        batch_size: int = 128,
        seq_len: int = 5,
        train_epochs: int = 10,
        train_verbose: bool = False,
        threshold_fn="sigmoid",
        hidden_size=None,
        hidden_mult=None,
        ensemble_th=1.05,
        base_learner_weights=None,
        output_size=2,
        multi_head=True,
        ignore_option=True,
        expand_option=True,
        checkpoint_freq=None,
        drift_delay=None,
        grace_period=None,
        data_points_previous_concept=1000,
        minimum_k_difference=0.04,
    ):
        """
        Parameters
        ----------
        model_class : class, default: PiggyBackGRU
            The network architecture class to instantiate.
        device : str, default: None
            The processing device ('cpu' or 'cuda').
        ensemble_batches : int, default: 30
            The number of batches to wait before choosing the best model from the ensemble
            after a drift detection.
        lr : float, default: 0.01
            Learning rate for the optimizer.
        input_size : int, default: 4
            Dimensionality of the input features.
        batch_size : int, default: 128
            Number of samples per mini-batch.
        seq_len : int, default: 5
            Length of the sequence sliding window.
        train_epochs : int, default: 10
            Number of training epochs per mini-batch.
        train_verbose : bool, default: False
            If True, prints epoch-level training metrics.
        threshold_fn : str, default: "sigmoid"
            The masking threshold function (e.g., "sigmoid", "binarizer").
        hidden_size : int, default: None
            Number of hidden units in the recurrent layer.
        hidden_mult : int, default: None
            Number of units to add during architectural expansion (accommodation).
        ensemble_th : float, default: 1.05
            Performance threshold multiplier; expansion is chosen if its performance
            is greater than mask-only performance multiplied by this factor.
        base_learner_weights : dict, default: None
            Pre-initialized weights to inject into the base learner.
        output_size : int, default: 2
            Number of output classes.
        multi_head : bool, default: True
            Whether to use an independent linear head for each new concept.
        ignore_option : bool, default: True
            Whether to include an 'Ignore' candidate in the ensemble for false alarm robustness.
        expand_option : bool, default: True
            Whether to include an 'Expand' candidate in the ensemble for structural accommodation.
        checkpoint_freq : int, default: None
            Frequency (in data points) to save model checkpoints.
        drift_delay : int, default: None
            Maximum expected delay of the drift detector, used for checkpoint retrieval.
        grace_period : int, default: None
            Number of data points to ignore new drift alarms after an adaptation.
        data_points_previous_concept : int, default: 1000
            Maximum number of labeled samples used to check forgetting after Ignore.
        minimum_k_difference : float, default: 0.04
            Minimum additive Kappa advantage for retaining the pre-drift model.
        """
        self.device = device
        self.model_class = model_class
        self.input_size = input_size
        self.lr = lr
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.train_verbose = train_verbose
        self.data_point_counter = 0
        self.multi_head = multi_head
        self.ignore_option = ignore_option
        self.expand_option = expand_option
        self.checkpoint_freq = checkpoint_freq
        self.checkpoints = []
        self.grace_period = grace_period
        self.previous_piggy = False
        # Consider PiggyMask_last only after the first PiggyMask choice

        self.threshold_fn = threshold_fn
        self.drift_delay = drift_delay

        # Ensemble parameters
        self.ensemble = {}
        self.ensemble_masks = {}
        self.ensemble_optims = {}
        self.ensemble_batches = ensemble_batches
        self.ensemble_counter = 0
        self.ensemble_th = ensemble_th
        self.ensemble_choices = []
        self.ensemble_perf = []
        self.biases = {}
        self.output_size = output_size

        self.current_perf = {}
        self.in_expansion = False
        self.piggymask_list = {}  # Piggymasks of previous tasks
        self.mask_list = {}  # Masks used for freezing parameters
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.criterion.to(self.device)
        self.hidden_mult = hidden_mult
        self.curr_task_idx = 1
        self.initial_task_id = 1
        self.training_counter = 0
        self.last_pred = {}
        self.pending_predictions = {}
        self.previous_model_state = None
        self.forgotten_models = {}
        self._initialize_concept_memory(
            data_points_previous_concept, minimum_k_difference
        )
        self.only_ignore = False
        self.in_grace_period = grace_period is not None and grace_period > 0
        self.grace_period_counter = 0

        if model_class == PiggyBackGRU:
            self.model = PiggyBackGRU(
                input_size=self.input_size,
                device=self.device,
                hidden_size=hidden_size,
                output_size=self.output_size,
                threshold_fn=threshold_fn,
                seq_len=seq_len,
                cGRU_weights=base_learner_weights,
                multi_head=multi_head,
            )
            self.model.train()

        # Freeze the piggymasks for the first task since we learn all weights.
        for name, param in self.model.named_parameters():
            if name.__contains__("mask"):
                param.requires_grad = False

        self.mask_list[1] = self.model.create_freezemask()

        self.model_optim = self._create_optimizer(self.model)

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Older pickle checkpoints did not retain timestamped predictions.
        self.__dict__.setdefault("pending_predictions", {})
        if self.grace_period is None or self.grace_period <= 0:
            self.in_grace_period = False
        self.__dict__.setdefault("grace_period_counter", 0)
        self.__dict__.setdefault("data_points_previous_concept", 1000)
        self.__dict__.setdefault("minimum_k_difference", 0.04)
        self.__dict__.setdefault("concept_x", {"old": [], "new": []})
        self.__dict__.setdefault("concept_y", {"old": [], "new": []})
        self.__dict__.setdefault("historical_heads", {})

    def _initialize_concept_memory(
        self, data_points_previous_concept, minimum_k_difference
    ):
        self.data_points_previous_concept = data_points_previous_concept
        self.minimum_k_difference = minimum_k_difference
        self.concept_x = {"old": [], "new": []}
        self.concept_y = {"old": [], "new": []}
        self.historical_heads = {}

    def collect_concept_sample(self, x, y):
        if len(self.concept_x["new"]) < self.data_points_previous_concept:
            self.concept_x["new"].append(np.asarray(x).copy())
            self.concept_y["new"].append(int(y))

    def _start_concept_sample(self):
        self.concept_x["old"] = self.concept_x["new"]
        self.concept_y["old"] = self.concept_y["new"]
        self.concept_x["new"] = []
        self.concept_y["new"] = []
        if self.multi_head:
            task_id = self.curr_task_idx - 1
            self.historical_heads[task_id] = pickle.loads(
                pickle.dumps(self.previous_model_state.heads[str(task_id)])
            )

    def _score_concept_model(self, model, task_id):
        # Only copies are evaluated: neither temporal state nor optimizer of the
        # online model participates in this forgetting check.
        model = pickle.loads(pickle.dumps(model))
        model.eval()
        score = metrics.CohenKappa()
        sample = np.asarray(self.concept_x["old"], dtype=np.float32)
        with torch.no_grad():
            for i, y in enumerate(self.concept_y["old"]):
                pred = 0
                if i >= self.seq_len - 1:
                    sequence = torch.as_tensor(
                        sample[i - self.seq_len + 1 : i + 1],
                        device=self.device,
                    ).unsqueeze(0)
                    outputs = model(sequence, task_id=task_id)
                    pred = int(get_pred_from_outputs(outputs)[0][-1].item())
                score.update(y, pred)
        return score.get()

    def _maybe_archive_previous_model(self, current_model):
        old_k = self._score_concept_model(
            self.previous_model_state, self.curr_task_idx - 1
        )
        current_k = self._score_concept_model(current_model, self.curr_task_idx)
        improvement = old_k - current_k
        retain = improvement > 0 and improvement >= self.minimum_k_difference
        if retain:
            self.forgotten_models[self.curr_task_idx - 1] = pickle.loads(
                pickle.dumps(self.previous_model_state)
            )
        return {
            "old_model": retain,
            "old_model_k": old_k,
            "current_model_k": current_k,
            "previous_concept_sample_size": len(self.concept_y["old"]),
        }

    def restore_historical_head(self, task_id):
        if self.multi_head and task_id in self.historical_heads:
            self.model.heads[str(task_id)] = pickle.loads(
                pickle.dumps(self.historical_heads[task_id])
            ).to(self.device)

    def _create_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def store_masks_biases(self):
        if self.in_expansion:
            self._decide_best_model()

        self.piggymask_list[self.curr_task_idx] = pickle.loads(
            pickle.dumps(self.model.get_piggymasks())
        )
        if not self.multi_head:
            self.biases[self.curr_task_idx] = pickle.loads(
                pickle.dumps(self.model.classifier[1].bias)
            )
        else:
            self.biases[self.curr_task_idx] = None
        if self.in_expansion:
            self._decide_best_model()

    def add_new_column(self, task_id):
        """
        Triggers the drift adaptation mechanism (ensemble phase).

        It freezes current weights, stores the past configuration, and spawns
        ensemble candidates (Mask, Expand, Ignore) to handle the new concept.

        Parameters
        ----------
        task_id : int
            The identifier for the new concept/task.
        """
        # Freeze past by setting freeze masks to 1s
        if self.in_expansion:
            self.ensemble_choices.append(
                {
                    "training_counter": self.training_counter,
                    "data_point_counter": self.data_point_counter,
                    "choice": "already_in_expansion",
                    "current_perf": None,
                    "chosen_model_size": None,
                }
            )
            self.ensemble_choices = sorted(
                self.ensemble_choices, key=lambda x: x["data_point_counter"]
            )
            print("MAGIC Net is already in expansion")
            return

        if self.in_grace_period:
            self.ensemble_choices.append(
                {
                    "training_counter": self.training_counter,
                    "data_point_counter": self.data_point_counter,
                    "choice": "grace_period",
                    "current_perf": None,
                    "chosen_model_size": None,
                }
            )
            self.ensemble_choices = sorted(
                self.ensemble_choices, key=lambda x: x["data_point_counter"]
            )
            print("MAGIC Net is in grace period")
            return
        self.ensemble_perf.append([])

        self.in_expansion = True
        self.ensemble_counter = 0

        # Store the model for the ignore option
        ignore_model = pickle.loads(pickle.dumps(self.model))
        if self.multi_head:
            ignore_model.add_task_head(task_id, previous=True)
        ignore_masks = pickle.loads(pickle.dumps(self.mask_list[task_id - 1]))

        # Restore previous state
        backbone_model = self.get_previous_state()
        self.checkpoints = []
        self.previous_model_state = pickle.loads(pickle.dumps(backbone_model))
        self.curr_task_idx = task_id
        self._start_concept_sample()

        # Freeze the past
        freeze_all_masks = pickle.loads(pickle.dumps(self.mask_list[task_id - 1]))
        for mask in freeze_all_masks.values():
            mask.fill_(1)
        self.piggymask_list[task_id - 1] = pickle.loads(
            pickle.dumps(backbone_model.get_piggymasks())
        )
        if not self.multi_head:
            self.biases[task_id - 1] = pickle.loads(
                pickle.dumps(backbone_model.classifier[1].bias)
            )
        else:
            self.biases[task_id - 1] = None
        if self.multi_head:
            backbone_model.add_task_head(task_id)

        backbone_piggy_last = pickle.loads(pickle.dumps(backbone_model))
        # Reinitialize piggymasks
        backbone_model.reinit_piggymask(
            freeze_masks=freeze_all_masks,
        )

        # Create ensemble:
        # "PiggyMask" stays the same and we just train the piggymask
        # "Expand" expands the hidden size

        self.ensemble["PiggyMask"] = pickle.loads(pickle.dumps(backbone_model))
        self.ensemble["PiggyMask"].reinit_linear_bias()
        self.ensemble_masks["PiggyMask"] = pickle.loads(pickle.dumps(freeze_all_masks))
        self.current_perf["PiggyMask"] = metrics.CohenKappa()
        self.ensemble_optims["PiggyMask"] = self._create_optimizer(
            self.ensemble["PiggyMask"]
        )

        if self.expand_option:
            self.ensemble["Expand"] = pickle.loads(pickle.dumps(backbone_model))
            self.ensemble_masks["Expand"] = self.ensemble["Expand"].expand_hidden(
                self.hidden_mult
            )
            self.current_perf["Expand"] = metrics.CohenKappa()
            self.ensemble_optims["Expand"] = self._create_optimizer(
                self.ensemble["Expand"]
            )

        # From the third concept the ensemble is enlarged based on the mode chosen

        if self.previous_piggy:
            backbone_piggy_last.reinit_newly_frozen_masks(ignore_masks)
            self.ensemble["PiggyMask_last"] = backbone_piggy_last
            # Reinitialize piggymasks with the ones from the previous concept
            self.ensemble_masks["PiggyMask_last"] = pickle.loads(
                pickle.dumps(freeze_all_masks)
            )
            self.current_perf["PiggyMask_last"] = metrics.CohenKappa()
            self.ensemble_optims["PiggyMask_last"] = self._create_optimizer(
                self.ensemble["PiggyMask_last"]
            )

        if self.ignore_option:
            self.ensemble["Ignore"] = ignore_model
            self.ensemble_masks["Ignore"] = ignore_masks
            self.current_perf["Ignore"] = metrics.CohenKappa()
            self.ensemble_optims["Ignore"] = self._create_optimizer(
                self.ensemble["Ignore"]
            )

        for model in self.ensemble.values():
            model.to(self.device)

        self.last_pred = {}
        self.pending_predictions = {}

    def get_previous_state(self):
        if len(self.checkpoints) > 0:
            models = [
                sc
                for sc in self.checkpoints
                if sc["data_point_counter"]
                <= self.data_point_counter - self.drift_delay
            ]
            if len(models) > 0:
                self.checkpoints = []
                return pickle.loads(pickle.dumps(models[-1]["model"]))
        return pickle.loads(pickle.dumps(self.model))

    def force_decision(self):
        if not self.in_expansion:
            return
        self._decide_best_model()

    def _decide_best_model(self):
        for name, item in self.current_perf.items():
            print(name, " performance: ", item)

        non_expand_options = ["PiggyMask"]
        if self.ignore_option:
            non_expand_options.append("Ignore")
        if self.previous_piggy:
            non_expand_options.append("PiggyMask_last")
        bestpiggy = max(non_expand_options, key=lambda k: self.current_perf[k].get())

        if self.expand_option:
            expand_options = ["Expand"]
            bestexpand = max(expand_options, key=lambda k: self.current_perf[k].get())
            if self.current_perf[bestexpand].get() > max(
                self.current_perf[bestpiggy].get(),
                self.ensemble_th * self.current_perf[bestpiggy].get(),
            ):
                best_model = bestexpand
            else:
                best_model = bestpiggy
        else:
            best_model = bestpiggy

        retention = {}
        if best_model != "Ignore":
            self.previous_piggy = True
            self.in_grace_period = (
                self.grace_period is not None and self.grace_period > 0
            )
            self.grace_period_counter = 0
        else:
            retention = self._maybe_archive_previous_model(self.ensemble[best_model])

        self.ensemble_choices.append(
            {
                "training_counter": self.training_counter - self.ensemble_batches,
                "data_point_counter": self.data_point_counter
                - self.ensemble_batches * self.batch_size,
                "choice": best_model,
                "final_perf": {
                    k: self.current_perf[k].get() for k in self.current_perf
                },
                "chosen_model_size": compute_model_size(self.ensemble[best_model]),
                **retention,
            }
        )
        self.ensemble_choices = sorted(
            self.ensemble_choices, key=lambda x: x["data_point_counter"]
        )
        print("Magic Net choses: ", best_model)
        self.model = self.ensemble[best_model]
        self.model_optim = self.ensemble_optims[best_model]
        self.mask_list[self.curr_task_idx] = pickle.loads(
            pickle.dumps(self.ensemble_masks[best_model])
        )
        self.in_expansion = False
        self.concept_x["old"] = []
        self.concept_y["old"] = []
        self.previous_model_state = None
        if self.checkpoint_freq is not None:
            self.save_checkpoint()
        self.ensemble = {}
        self.ensemble_masks = {}
        self.ensemble_optims = {}
        self.current_perf = {}
        self.last_pred = {}
        self.pending_predictions = {}

    def manage_grace_period(self):
        if self.in_grace_period:
            self.grace_period_counter += 1
            if self.grace_period_counter >= self.grace_period:
                self.grace_period_counter = 0
                self.in_grace_period = False
                print("MAGIC Net cancels the grace period")

    def update_counter(self, counter, advance_grace_period=True):
        self.data_point_counter = counter
        if self.checkpoint_freq is not None and not self.in_expansion:
            if self.data_point_counter % self.checkpoint_freq == 0:
                self.save_checkpoint()
        if advance_grace_period:
            self.manage_grace_period()

    def train(self, x, y):
        """
        Executes a training step on a given mini-batch. If the model is in expansion mode,
        it trains the entire ensemble in parallel.

        Parameters
        ----------
        x : torch.Tensor
            Batch input features tensor.
        y : torch.Tensor
            Batch target labels tensor.

        Returns
        -------
        dict
            Dictionary containing historical training metrics across epochs
            (accuracy, loss, kappa).

        """
        perf_train = {
            "accuracy": [],
            "loss": [],
            "kappa": [],
            "kappa_temporal": [],
        }

        for e in range(1, self.train_epochs + 1):
            if not self.in_expansion:
                perf_epoch = self._fit(x, y)
            else:
                perf_epoch = self._fit_ensemble(x, y)

            if self.train_verbose:
                print(
                    "Task ",
                    self.curr_task_idx,
                    "Training epoch ",
                    e,
                    "/",
                    self.train_epochs,
                    ". accuracy: ",
                    perf_epoch["accuracy"],
                    ", loss:",
                    perf_epoch["loss"],
                    sep=" ",
                    end="\r",
                )
            for k in perf_epoch:
                perf_train[k].append(perf_epoch[k])

        if self.train_verbose:
            print()
            print()
        if self.in_expansion:
            self.ensemble_counter = self.ensemble_counter + 1
            self.ensemble_perf[-1].append(
                {k: self.current_perf[k].get() for k in self.current_perf}
            )
            if self.ensemble_counter == self.ensemble_batches:
                self._decide_best_model()
        self.training_counter += 1
        return perf_train

    def save_checkpoint(self):
        if self.checkpoint_freq is not None:
            self.checkpoints.append(
                {
                    "data_point_counter": self.data_point_counter,
                    "training_counter": self.training_counter,
                    "model": pickle.loads(pickle.dumps(self.model)),
                }
            )
            max_keep = int((self.drift_delay / self.checkpoint_freq) + 2)
            max_keep = max(4, max_keep)
            self.checkpoints = self.checkpoints[-max_keep:]

    def _fit(self, x, y):
        x, y = x.to(self.model.device), y.to(self.model.device)
        outputs = self.model(x, task_id=self.curr_task_idx)
        """if not self.loss_on_seq:
            outputs = get_samples_outputs(outputs)"""

        loss = customized_loss(outputs, y, self.criterion, self.model.device)

        self.model.train()
        self.model_optim.zero_grad()
        loss.backward()
        if self.curr_task_idx != 1:
            self.delete_gradients(self.model, self.mask_list[self.curr_task_idx])

        self.model_optim.step()
        outputs = self.model(x, task_id=self.curr_task_idx)

        perf_train = {
            "loss": loss.item(),
            "accuracy": accuracy(outputs, y).item(),
            "kappa": cohen_kappa(outputs, y, device=self.device).item(),
        }
        return perf_train

    def _fit_ensemble(self, x, y):
        temp_loss = {}
        temp_accuracy = {}
        temp_kappa = {}

        for name, model in self.ensemble.items():
            x_i, y_i = x.to(model.device), y.to(model.device)
            model.train()
            outputs = model(x_i, task_id=self.curr_task_idx)

            loss = customized_loss(outputs, y_i, self.criterion, device=self.device)

            self.ensemble_optims[name].zero_grad()
            loss.backward()

            self.delete_gradients(model, self.ensemble_masks[name])

            self.ensemble_optims[name].step()
            outputs = model(x_i, task_id=self.curr_task_idx)

            temp_loss.update({name: loss.item()})
            temp_accuracy.update({name: accuracy(outputs, y_i).item()})
            temp_kappa.update(
                {name: cohen_kappa(outputs, y_i, device=model.device).item()}
            )

        perf_train = {"loss": temp_loss, "accuracy": temp_accuracy, "kappa": temp_kappa}
        return perf_train

    def delete_gradients(self, model, masks):
        """
        It sets the computed gradient for previous parameters of a given model to 0, effectively freezing them.
        """
        if not masks:
            return

        for name, param in model.classifier.named_parameters():
            if param.grad is None:
                continue

            if name[2:] in masks:
                mask = masks[name[2:]]
                param.grad.data[mask.ne(0)] = 0
            if name.__contains__("mask"):
                key = name[12:]
                if key in masks:
                    mask = masks[key]
                    param.grad.data[mask.eq(0)] = 0
                else:
                    print(key)

    def predict_one(self, x, task_id=None, timestamp=-1):
        """
        Computes the model's prediction on the given input. If MAGIC Net is in ensemble phase it returns the prediction
        of the current best-performing model.
        """
        if task_id is None:
            task_id = self.curr_task_idx
        if self.in_expansion is False:
            pred = self.model(x, task_id=task_id)
            return pred

        for name, model in self.ensemble.items():
            self.last_pred[name] = model(x, task_id=task_id)

        # Store only class labels: delayed evaluation needs neither the graph
        # nor the full logits. The -1 slot preserves immediate-label usage.
        self.pending_predictions[timestamp] = {
            name: int(get_pred_from_outputs(pred)[0][-1].detach().cpu().item())
            for name, pred in self.last_pred.items()
        }

        best_model = max(zip(self.current_perf.values(), self.current_perf.keys()))[1]
        pred = self.last_pred[best_model]

        return pred

    def update_metrics(self, y, timestamp=-1):
        # Labels for pre-drift points or sequence warm-up have no candidate
        # prediction. Popping also prevents scoring the same label twice.
        predictions = self.pending_predictions.pop(timestamp, {})
        for name, pred in predictions.items():
            self.current_perf[name].update(y, pred)

    def get_state_dict(self):
        if self.in_expansion is False:
            return self.model.state_dict()
        else:
            return None
