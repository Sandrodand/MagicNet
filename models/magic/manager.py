from river import metrics
import copy
import pickle

from evaluation.evaluation_utils import compute_model_size
from models.magic.piggyback_cgru import (
    PiggyBackGRU,
)
from models.utils_scl.utils import *


class MagicManager(object):
    """Handles training and pruning."""

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
        initial_task_id=1,
        mask_init="1s",
        hidden_size=None,
        hidden_mult=None,
        ensemble_th=None,
        cgru_weights=None,
        cap_sigmoid=True,
        output_size=2,
        multi_head=True,
        ignore_option=True,
        checkpoint_freq=None,
        drift_delay=None,
        grace_period=None
    ):
        self.device = device
        self.model_class = model_class
        self.mask_init = mask_init
        self.input_size = input_size
        self.lr = lr
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.train_verbose = train_verbose
        self.data_point_counter = 0
        self.multi_head = multi_head
        self.ignore_option = ignore_option
        self.checkpoint_freq = checkpoint_freq
        self.checkpoints = []
        self.grace_period = grace_period

        self.threshold_fn = threshold_fn
        self.drift_delay=drift_delay

        # Ensemble parameters
        self.ensemble = {}
        self.ensemble_masks = {}
        self.ensemble_optims = {}
        self.ensemble_batches = ensemble_batches
        self.ensemble_counter = 0
        self.ensemble_th = ensemble_th
        self.ensemble_choices = []
        self.ensemble_perf = []
        self.biases = []
        self.cap_sigmoid = cap_sigmoid
        self.output_size = output_size

        self.current_perf = {}
        self.in_expansion = False
        self.piggymask_list = [] # Piggymasks of previous tasks
        self.mask_list = {} # Masks used for freezing parameters
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.criterion.to(self.device)
        self.hidden_mult = hidden_mult
        self.curr_task_idx = initial_task_id
        self.initial_task_id = initial_task_id
        self.training_counter = 0
        self.last_pred = {}
        self.previous_piggy = False
        self.previous_model_state = None
        self.forgotten_models = {}
        if grace_period is not None:
            self.in_grace_period = True
            self.grace_period_counter = 0

        if model_class == PiggyBackGRU:
            self.model = PiggyBackGRU(
                input_size=self.input_size,
                device=self.device,
                hidden_size=hidden_size,
                mask_init=mask_init,
                threshold_fn=threshold_fn,
                seq_len=seq_len,
                cGRU_weights=cgru_weights,
                cap_sigmoid=self.cap_sigmoid,
                output_size=self.output_size,
                initial_task_id=initial_task_id,
                multi_head=multi_head
            )
            self.model.train()

        if initial_task_id == 1:
            # Freeze the piggymasks for the first task since we are learning all of the weights
            for name, param in self.model.named_parameters():
                if name.__contains__("mask"):
                    param.requires_grad = False

            # Create initial freeze masks
            self.mask_list[initial_task_id] = self.model.create_freezemask()

        self.model_optim = self._create_optimizer(self.model)

    def _create_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def store_masks_biases(self):
        self.piggymask_list.append(self.model.get_piggymasks())
        if not self.multi_head:
            self.biases.append(pickle.loads(pickle.dumps(self.model.classifier[1].bias)))
        else:
            self.biases.append(None)
        if self.in_expansion:
            self._decide_best_model()

    def add_new_column(self, task_id):
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
        current_model_copy = copy.deepcopy(self.model)
        if self.multi_head:
            current_model_copy.add_task_head(task_id, previous=True)
        current_freeze_mask = copy.deepcopy(self.mask_list[task_id - 1])

        # Restore previous state
        self.model = self.get_previous_state()
        self.previous_model_state = copy.deepcopy(self.model)
        self.curr_task_idx = task_id

        # Freeze the past
        for mask in self.mask_list[task_id - 1].values():
            mask.fill_(1)
        self.piggymask_list.append(self.model.get_piggymasks())
        if not self.multi_head:
            self.biases.append(pickle.loads(pickle.dumps(self.model.classifier[1].bias)))
        else:
            self.biases.append(None)
        if self.multi_head:
            self.model.add_task_head(task_id)

        # Reinitialize piggymasks
        self.model.reinit_piggymask(
            self.mask_init, freeze_masks=self.mask_list[task_id - 1]
        )

        # Create ensemble:
        # "PiggyMask" stays the same and we just train the piggymask
        # "Expand" expands the hidden size

        self.ensemble["PiggyMask"] = copy.deepcopy(self.model)
        self.ensemble["PiggyMask"].reinit_linear_bias()
        self.ensemble_masks["PiggyMask"] = self.mask_list[task_id - 1]
        self.current_perf["PiggyMask"] = metrics.CohenKappa()
        self.ensemble_optims["PiggyMask"] = self._create_optimizer(
            self.ensemble["PiggyMask"]
        )

        self.ensemble["Expand"] = copy.deepcopy(self.model)
        self.ensemble_masks["Expand"] = self.ensemble["Expand"].expand_hidden(
            self.hidden_mult
        )
        self.current_perf["Expand"] = metrics.CohenKappa()
        self.ensemble_optims["Expand"] = self._create_optimizer(self.ensemble["Expand"])

        # From the third concept the ensemble is enlarged based on the mode chosen

        if self.previous_piggy:
            self.ensemble["PiggyMask_last"] = copy.deepcopy(self.ensemble["PiggyMask"])
            # Reinitialize piggymasks with the ones from the previous concept
            self.ensemble["PiggyMask_last"].reinit_piggymask(
                self.mask_init,
                freeze_masks=self.mask_list[task_id - 1],
                masks=self.piggymask_list[-1],
            )
            self.ensemble_masks["PiggyMask_last"] = self.mask_list[task_id - 1]
            self.current_perf["PiggyMask_last"] = metrics.CohenKappa()
            self.ensemble_optims["PiggyMask_last"] = self._create_optimizer(
                self.ensemble["PiggyMask_last"]
            )

        if self.ignore_option:
            self.ensemble["Ignore"] = current_model_copy
            self.ensemble_masks["Ignore"] = current_freeze_mask
            self.current_perf["Ignore"] = metrics.CohenKappa()
            self.ensemble_optims["Ignore"] = self._create_optimizer(self.ensemble["Ignore"])

        for model in self.ensemble.values():
            model.to(self.device)
        self.last_pred = {}

    def get_previous_state(self):
        if len(self.checkpoints) > 0:
            models = [
                sc
                for sc in self.checkpoints
                if sc["data_point_counter"] <= self.data_point_counter - self.drift_delay
            ]
            if len(models) > 0:
                self.checkpoints = []
                return models[-1]["model"]
        return copy.deepcopy(self.model)

    def _decide_best_model(self):
        for name, item in self.current_perf.items():
            print(name, " performance: ", item)

        non_expand_options = ["PiggyMask"]
        if self.ignore_option:
            non_expand_options.append("Ignore")
        if self.previous_piggy:
            non_expand_options.append("PiggyMask_last")
        bestpiggy = max(non_expand_options, key=lambda k: self.current_perf[k].get())

        expand_options = ["Expand"]
        bestexpand = max(expand_options, key=lambda k: self.current_perf[k].get())

        if self.current_perf[bestexpand].get() > self.ensemble_th * self.current_perf[bestpiggy].get():
            best_model = bestexpand
        else:
            best_model = bestpiggy

        if best_model != "Ignore":
            self.previous_piggy = True
            self.in_grace_period = True
            self.grace_period_counter = 0
        else:
            self.forgotten_models[self.curr_task_idx-1] = copy.deepcopy(self.previous_model_state)

        self.ensemble_choices.append(
            {
                "training_counter": self.training_counter - self.ensemble_batches,
                "data_point_counter": self.data_point_counter - self.ensemble_batches * self.batch_size,
                "choice": best_model,
                "final_perf": {
                    k: self.current_perf[k].get() for k in self.current_perf
                },
                "chosen_model_size": compute_model_size(self.ensemble[best_model]),
            }
        )
        self.ensemble_choices = sorted(self.ensemble_choices, key=lambda x: x["data_point_counter"])
        print("Magic Net choses: ", best_model)
        self.model = self.ensemble[best_model]
        self.model_optim = self.ensemble_optims[best_model]
        self.mask_list[self.curr_task_idx] = self.ensemble_masks[best_model]
        self.in_expansion = False
        self.save_checkpoint()
        self.ensemble = {}
        self.ensemble_masks = {}
        self.ensemble_optims = {}
        self.current_perf = {}

    def manage_grace_period(self):
        if self.in_grace_period:
            self.grace_period_counter += 1
            if self.grace_period_counter >= self.grace_period:
                self.grace_period_counter = 0
                self.in_grace_period = False
                print("MAGIC Net cancels the grace period")

    def update_counter(self, counter):
        self.data_point_counter = counter
        self.manage_grace_period()

    def train(self, x, y, counter=None):
        self.data_point_counter = counter
        self.manage_grace_period()
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
        if self.checkpoint_freq is not None and not self.in_expansion:
            if self.data_point_counter % self.checkpoint_freq == 0:
                self.save_checkpoint()
        return perf_train

    def save_checkpoint(self):
        if self.checkpoint_freq is not None:
            self.checkpoints.append({
                "data_point_counter": self.data_point_counter,
                "training_counter": self.training_counter,
                "model": copy.deepcopy(self.model)
            })
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

    def predict_one(self, x, task_id=None):
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

        best_model = max(zip(self.current_perf.values(), self.current_perf.keys()))[1]
        pred = self.last_pred[best_model]

        return pred

    def update_metrics(self, y):
        for name, pred in self.last_pred.items():
            with torch.no_grad():
                _pred, _ = get_pred_from_outputs(pred)
                _pred = int(_pred[-1].detach().cpu().numpy())
            self.current_perf[name].update(y, _pred)

    def get_state_dict(self):
        if self.in_expansion is False:
            return self.model.state_dict()
        else:
            return None
