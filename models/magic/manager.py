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
        ensemble_batches=50,
        lr=0.01,
        input_size=4,
        batch_size: int = 128,
        seq_len: int = 5,
        train_epochs: int = 10,
        train_verbose: bool = False,
        threshold_fn="sigmoid",
        initial_task_id=1,
        mask_init="1s",
        hidden_size=50,
        hidden_mult=50,
        ensemble_th=1.1,
        cGRU_weights=None,
        ensemble_mode="last",
        cap_sigmoid=True,
        expand_last=False,
        output_size=2
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

        self.threshold_fn = threshold_fn

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

        self.ensemble_mode = ensemble_mode
        self.current_perf = {}
        self.in_expansion = False
        self.piggymask_list = [] # Piggymasks of previous tasks
        self.mask_list = {} # Masks used for freezing parameters
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.criterion.to(self.device)
        self.hidden_mult = hidden_mult
        self.curr_task_idx = initial_task_id
        self.counter = 0
        self.last_pred = {}
        self.expand_last = expand_last

        if model_class == PiggyBackGRU:
            self.model = PiggyBackGRU(
                input_size=self.input_size,
                device=self.device,
                hidden_size=hidden_size,
                mask_init=mask_init,
                threshold_fn=threshold_fn,
                seq_len=seq_len,
                cGRU_weights=cGRU_weights,
                cap_sigmoid=self.cap_sigmoid,
                output_size=self.output_size
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
        self.biases.append(pickle.loads(pickle.dumps(self.model.classifier[1].bias)))
        if self.in_expansion:
            self._decide_best_model()

    def add_new_column(self, task_id):
        # Freeze past by setting freeze masks to 1s
        if self.in_expansion:
            self.ensemble_choices.append(
                {
                    "cont": self.counter,
                    "choice": "ignore",
                    "current_perf": None,
                    "chosen_model_size": None,
                }
            )
            self.ensemble_choices = sorted(
                self.ensemble_choices, key=lambda x: x["cont"]
            )
            return
        self.ensemble_perf.append([])
        self.curr_task_idx = task_id
        for mask in self.mask_list[task_id - 1].values():
            mask.fill_(1)
        if task_id > 2:
            self.piggymask_list.append(self.model.get_piggymasks())
        else:
            self.piggymask_list.append([])
        self.biases.append(pickle.loads(pickle.dumps(self.model.classifier[1].bias)))
        self.in_expansion = True
        self.ensemble_counter = 0

        # Reinitialize piggymasks
        self.model.reinit_piggymask(
            self.mask_init, freeze_masks=self.mask_list[task_id - 1]
        )

        # Create ensemble:
        # "PiggyMask" stays the same and we just train the piggymask
        # "Expand" expands the hidden size

        self.ensemble["PiggyMask"] = self.model
        self.ensemble["PiggyMask"].reinit_linear_bias()
        self.ensemble["Expand"] = PiggyBackGRU(
            input_size=self.input_size,
            device=self.device,
            hidden_size=self.model.hidden_size,
            mask_init=self.mask_init,
            threshold_fn=self.threshold_fn,
            seq_len=self.seq_len,
            cap_sigmoid=self.cap_sigmoid,
            output_size=self.output_size
        )
        self.ensemble["Expand"].load_state_dict(self.model.state_dict())

        self.ensemble_masks["PiggyMask"] = self.mask_list[task_id - 1]
        self.ensemble_masks["Expand"] = self.ensemble["Expand"].expand_hidden(
            self.hidden_mult
        )

        self.current_perf["PiggyMask"] = metrics.CohenKappa()
        self.current_perf["Expand"] = metrics.CohenKappa()

        self.ensemble_optims["PiggyMask"] = self._create_optimizer(
            self.ensemble["PiggyMask"]
        )
        self.ensemble_optims["Expand"] = self._create_optimizer(self.ensemble["Expand"])

        # From the third concept the ensemble is enlarged based on the mode chosen

        if self.ensemble_mode != "classic" and task_id > 2:
            self.ensemble["PiggyMask_last"] = copy.deepcopy(self.ensemble["PiggyMask"])
            if self.ensemble_mode == "last":
                # Reinitialize piggymasks with the ones from the previous concept
                self.ensemble["PiggyMask_last"].reinit_piggymask(
                    self.mask_init,
                    freeze_masks=self.mask_list[task_id - 1],
                    masks=self.piggymask_list[-1],
                )
            else:
                # Reinitialize piggymasks with the average of previously learned ones
                new_masks = self.average_masks(self.piggymask_list)
                self.ensemble["PiggyMask_last"].reinit_piggymask(
                    self.mask_init,
                    freeze_masks=self.mask_list[task_id - 1],
                    masks=new_masks,
                )
            if self.expand_last:
                self.ensemble["Expand_last"] = copy.deepcopy(
                    self.ensemble["PiggyMask_last"]
                )
                self.ensemble_masks["Expand_last"] = self.ensemble[
                    "Expand_last"
                ].expand_hidden(self.hidden_mult)
                self.current_perf["Expand_last"] = metrics.CohenKappa()
                self.ensemble_optims["Expand_last"] = self._create_optimizer(
                    self.ensemble["Expand_last"]
                )
            self.ensemble_masks["PiggyMask_last"] = self.mask_list[task_id - 1]
            self.current_perf["PiggyMask_last"] = metrics.CohenKappa()
            self.ensemble_optims["PiggyMask_last"] = self._create_optimizer(
                self.ensemble["PiggyMask_last"]
            )
        for model in self.ensemble.values():
            model.to(self.device)
        self.last_pred = {}

    def average_masks(self, masklist):
        summed = {}
        denom = {}
        for name, mask in masklist[-1].items():
            denom[name] = torch.zeros_like(mask)
            summed[name] = torch.zeros_like(mask)

        for i in reversed(masklist):
            for name, mask in summed.items():
                if len(i[name].size()) == 2:
                    mask[: i[name].size()[0], : i[name].size()[1]] = (
                        mask[: i[name].size()[0], : i[name].size()[1]] + i[name]
                    )
                    denom[name][: i[name].size()[0], : i[name].size()[1]] = denom[name][
                        : i[name].size()[0], : i[name].size()[1]
                    ] + torch.ones_like(i[name])
                else:
                    mask[: i[name].size()[0]] = mask[: i[name].size()[0]] + i[name]
                    denom[name][: i[name].size()[0]] = denom[name][
                        : i[name].size()[0]
                    ] + torch.ones_like(i[name])
        for name, mask in summed.items():
            summed[name] = torch.div(mask, denom[name])
        return summed

    def _decide_best_model(self):
        for name, item in self.current_perf.items():
            print(name, " performance: ", item)
        if self.ensemble_mode == "classic" or self.curr_task_idx <= 2:
            if (
                self.current_perf["Expand"].get()
                > self.ensemble_th * self.current_perf["PiggyMask"].get()
            ):
                best_model = "Expand"
            else:
                best_model = "PiggyMask"
        else:
            if (
                self.current_perf["PiggyMask"].get()
                > self.current_perf["PiggyMask_last"].get()
            ):
                bestpiggy = "PiggyMask"
            else:
                bestpiggy = "PiggyMask_last"
            bestexpand = "Expand"
            if self.expand_last:
                if (
                    self.current_perf["Expand"].get()
                    > self.current_perf["Expand_last"].get()
                ):
                    bestexpand = "Expand"
                else:
                    bestexpand = "Expand_last"
            if (
                self.current_perf[bestexpand].get()
                > self.ensemble_th * self.current_perf[bestpiggy].get()
            ):
                best_model = bestexpand
            else:
                best_model = bestpiggy
        self.ensemble_choices.append(
            {
                "cont": self.counter - self.ensemble_batches,
                "choice": best_model,
                "final_perf": {
                    k: self.current_perf[k].get() for k in self.current_perf
                },
                "chosen_model_size": compute_model_size(self.ensemble[best_model]),
            }
        )
        self.ensemble_choices = sorted(self.ensemble_choices, key=lambda x: x["cont"])
        print("CHOSEN MODEL: ", best_model)
        self.model = self.ensemble[best_model]
        self.model_optim = self.ensemble_optims[best_model]
        self.mask_list[self.curr_task_idx] = self.ensemble_masks[best_model]
        self.in_expansion = False
        self.ensemble = {}
        self.ensemble_masks = {}
        self.ensemble_optims = {}
        self.current_perf = {}

    def train(self, x, y):
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
        self.counter += 1
        return perf_train

    def _fit(self, x, y):
        x, y = x.to(self.model.device), y.to(self.model.device)
        outputs = self.model(x)
        """if not self.loss_on_seq:
            outputs = get_samples_outputs(outputs)"""

        loss = customized_loss(outputs, y, self.criterion, self.model.device)

        self.model.train()
        self.model_optim.zero_grad()
        loss.backward()
        if self.curr_task_idx != 1:
            self.delete_gradients(self.model, self.mask_list[self.curr_task_idx])

        self.model_optim.step()
        outputs = self.model(x)

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
            outputs = model(x_i)

            loss = customized_loss(outputs, y_i, self.criterion, device=self.device)

            self.ensemble_optims[name].zero_grad()
            loss.backward()

            self.delete_gradients(model, self.ensemble_masks[name])

            self.ensemble_optims[name].step()
            outputs = model(x_i)

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
        for name, param in model.classifier.named_parameters():
            if name[2:] in masks:
                mask = masks[name[2:]]
                param.grad.data[mask.ne(0)] = 0
            if name.__contains__("mask"):
                mask = masks[name[12:]]
                param.grad.data[mask.eq(0)] = 0

    def predict_one(self, x):
        """
        Computes the model's prediction on the given input. If MAGIC Net is in ensemble phase it returns the prediction
        of the current best-performing model.
        """
        if self.in_expansion is False:
            pred = self.model(x)
            return pred

        for name, model in self.ensemble.items():
            self.last_pred[name] = model(x)

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
