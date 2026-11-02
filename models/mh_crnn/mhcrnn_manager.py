"""MAGIC drift management without weight masks or hidden-size expansion."""

from copy import deepcopy

import torch
from river import metrics

from evaluation.evaluation_utils import compute_model_size
from models.crnn.cgru import cGRULinear
from models.mh_crnn.mhcrnn_model import MultiHeadCrnnModel
from models.magic.manager import MagicManager
from models.utils_scl.utils import accuracy, cohen_kappa, customized_loss


class MultiHeadCrnnManager(MagicManager):
    def __init__(
        self,
        model_class=cGRULinear,
        device="cpu",
        ensemble_batches=30,
        lr=0.01,
        input_size=4,
        batch_size=128,
        seq_len=5,
        train_epochs=10,
        train_verbose=False,
        hidden_size=50,
        ensemble_th=1.05,
        base_learner_weights=None,
        output_size=2,
        ignore_option=True,
        expand_option=True,
        checkpoint_freq=1500,
        drift_delay=3000,
        grace_period=5000,
        multi_head=True,
        data_points_previous_concept=1000,
        minimum_k_difference=0.04,
        **kwargs,
    ):
        if not multi_head:
            raise ValueError("The head-only ablation requires multi_head=True")
        if not ignore_option and not expand_option:
            raise ValueError("At least one drift candidate must be enabled")
        self.device = device
        self.model_class = model_class
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.train_verbose = train_verbose
        self.initial_task_id = 1
        self.curr_task_idx = 1
        self.multi_head = True
        self.ignore_option = ignore_option
        self.expand_option = expand_option
        self.checkpoint_freq = checkpoint_freq
        self.drift_delay = drift_delay
        self.checkpoints = []
        self.grace_period = grace_period
        self.in_grace_period = grace_period is not None and grace_period > 0
        self.grace_period_counter = 0
        self.data_point_counter = 0
        self.training_counter = 0
        self.in_expansion = False
        self.ensemble = {}
        self.ensemble_optims = {}
        self.ensemble_batches = ensemble_batches
        self.ensemble_th = ensemble_th
        self.ensemble_counter = 0
        self.ensemble_choices = []
        self.ensemble_perf = []
        self.current_perf = {}
        self.last_pred = {}
        self.pending_predictions = {}
        self.previous_model_state = None
        self.forgotten_models = {}
        self._initialize_concept_memory(
            data_points_previous_concept, minimum_k_difference
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.model = MultiHeadCrnnModel(
            model_class=model_class,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            batch_size=batch_size,
            device=device,
            base_learner_weights=base_learner_weights,
        )
        self.model_optim = self._create_optimizer(self.model)

    def _create_optimizer(self, model):
        return torch.optim.Adam(
            (param for param in model.parameters() if param.requires_grad), lr=self.lr
        )

    def restore_previous_state(self):
        """Retrieve the delayed checkpoint through MAGIC's existing mechanism."""
        return self.get_previous_state()

    def add_new_column(self, task_id):
        if self.in_expansion or self.in_grace_period:
            # Reuse MAGIC's alarm logging; it returns before touching masks.
            return super().add_new_column(task_id)

        ignore_model = deepcopy(self.model)
        self.previous_model_state = self.restore_previous_state()
        self.checkpoints = []
        self.curr_task_idx = task_id
        self._start_concept_sample()
        self.in_expansion = True
        self.ensemble_counter = 0
        self.ensemble_perf.append([])

        if self.expand_option:
            expand_model = deepcopy(self.previous_model_state)
            expand_model.freeze_backbone()
            expand_model.add_task_head(task_id)
            self.ensemble["Expand"] = expand_model
        if self.ignore_option:
            ignore_model.add_task_head(task_id, previous=True)
            self.ensemble["Ignore"] = ignore_model

        self.current_perf = {name: metrics.CohenKappa() for name in self.ensemble}
        self.ensemble_optims = {
            name: self._create_optimizer(model) for name, model in self.ensemble.items()
        }
        self.last_pred = {}
        self.pending_predictions = {}

    def _decide_best_model(self):
        if not self.in_expansion:
            return
        for name, item in self.current_perf.items():
            print(name, " performance: ", item)
        if "Ignore" not in self.ensemble:
            best_model = "Expand"
        elif "Expand" not in self.ensemble:
            best_model = "Ignore"
        elif self.current_perf["Expand"].get() > max(
            self.current_perf["Ignore"].get(),
            self.ensemble_th * self.current_perf["Ignore"].get(),
        ):
            best_model = "Expand"
        else:
            best_model = "Ignore"

        retention = {}
        if best_model == "Ignore":
            retention = self._maybe_archive_previous_model(self.ensemble[best_model])
        else:
            self.in_grace_period = (
                self.grace_period is not None and self.grace_period > 0
            )
            self.grace_period_counter = 0

        self.ensemble_choices.append(
            {
                "training_counter": self.training_counter - self.ensemble_batches,
                "data_point_counter": self.data_point_counter
                - self.ensemble_batches * self.batch_size,
                "choice": best_model,
                "final_perf": {k: v.get() for k, v in self.current_perf.items()},
                "chosen_model_size": compute_model_size(self.ensemble[best_model]),
                **retention,
            }
        )
        self.ensemble_choices.sort(key=lambda item: item["data_point_counter"])
        print("Head-only cRNN choses: ", best_model)
        self.model = self.ensemble[best_model]
        self.model_optim = self.ensemble_optims[best_model]
        self.in_expansion = False
        self.concept_x["old"] = []
        self.concept_y["old"] = []
        self.save_checkpoint()
        self.ensemble = {}
        self.ensemble_optims = {}
        self.current_perf = {}
        self.last_pred = {}
        self.pending_predictions = {}
        self.previous_model_state = None

    def _fit_model(self, model, optimizer, x, y):
        x, y = x.to(self.device), y.to(self.device)
        model.train()
        optimizer.zero_grad()
        outputs = model(x, task_id=self.curr_task_idx)
        loss = customized_loss(outputs, y, self.criterion, self.device)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            outputs = model(x, task_id=self.curr_task_idx)
            return {
                "loss": loss.item(),
                "accuracy": accuracy(outputs, y).item(),
                "kappa": cohen_kappa(outputs, y, device=self.device).item(),
            }

    def _fit(self, x, y):
        return self._fit_model(self.model, self.model_optim, x, y)

    def _fit_ensemble(self, x, y):
        results = {
            name: self._fit_model(model, self.ensemble_optims[name], x, y)
            for name, model in self.ensemble.items()
        }
        return {
            metric: {name: result[metric] for name, result in results.items()}
            for metric in ("loss", "accuracy", "kappa")
        }

    def store_masks_biases(self):
        """Compatibility with the prequential evaluator's finalization hook."""
        self.force_decision()
