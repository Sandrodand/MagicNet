"""Unmasked recurrent base learner with an independent head for each task."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from models.crnn.cgru import cGRULinear
from models.crnn.clstm import cLSTMLinear
from models.magic.piggyback_layers import GRUBlockMath, LSTMBlockMath


class MultiHeadCrnnModel(nn.Module):
    frozen_hidden_size = 0

    def __init__(
        self,
        model_class=cGRULinear,
        input_size=4,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        device="cpu",
        base_learner_weights=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        if base_learner_weights is None or isinstance(base_learner_weights, dict):
            self.learner = model_class(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                batch_size=batch_size,
                device=self.device,
            )
            if base_learner_weights is not None:
                self.learner.load_state_dict(base_learner_weights)
        else:
            # Use exactly the same initialized column as MAGIC Net and cPNN.
            if isinstance(base_learner_weights, (cGRULinear, cLSTMLinear)):
                self.learner = deepcopy(base_learner_weights)
            else:
                self.learner = deepcopy(base_learner_weights.columns.columns[0])
        if not isinstance(self.learner, (cGRULinear, cLSTMLinear)):
            raise TypeError("The base learner must be cGRULinear or cLSTMLinear")

        self.learner.device = self.device
        self.hidden_size = self.learner.hidden_size
        self.output_size = self.learner.output_size
        self.heads = nn.ModuleDict({"1": self.learner.linear})
        self.learner.linear = nn.Identity()
        self.backbone_frozen = False
        self.to(self.device)
        self.set_trainable_task(1)

    def forward(self, x, task_id=None):
        if task_id is None:
            task_id = list(self.heads)[-1]
        head = self.heads[str(task_id)]
        if head.in_features == self.hidden_size:
            _, hidden = self.learner(x, None)
        else:
            hidden = self._historical_hidden(x, head.in_features)
        return head(hidden[:, -1, :])

    def _historical_hidden(self, x, width):
        # Select the original width inside each gate before running the RNN.
        is_lstm = isinstance(self.learner, cLSTMLinear)
        recurrent = self.learner.lstm if is_lstm else self.learner.gru
        input_weights = recurrent.weight_ih_l0.view(
            -1, self.hidden_size, recurrent.input_size
        )
        recurrent_weights = recurrent.weight_hh_l0.view(
            -1, self.hidden_size, self.hidden_size
        )
        weights = (
            input_weights[:, :width].reshape(-1, recurrent.input_size).contiguous(),
            recurrent_weights[:, :width, :width].reshape(-1, width).contiguous(),
            recurrent.bias_ih_l0.view(-1, self.hidden_size)[:, :width].reshape(-1),
            recurrent.bias_hh_l0.view(-1, self.hidden_size)[:, :width].reshape(-1),
        )
        x = x.to(recurrent.weight_ih_l0)
        if is_lstm:
            h0 = self.learner._build_initial_state(x, self.learner.h0)
            c0 = self.learner._build_initial_state(x, self.learner.c0)
            state = (h0[..., :width].contiguous(), c0[..., :width].contiguous())
            return LSTMBlockMath(x, state, *weights, batch_first=True)
        h0 = self.learner._build_initial_state(x)[..., :width].contiguous()
        hidden, _ = GRUBlockMath(x, h0, *weights, batch_first=True)
        return hidden

    @torch.no_grad()
    def expand_hidden(self, hidden_mult, freeze_backbone):
        old_size = self.hidden_size
        new_size = old_size + hidden_mult
        recurrent_name = "lstm" if isinstance(self.learner, cLSTMLinear) else "gru"
        recurrent = getattr(self.learner, recurrent_name)
        expanded = type(recurrent)(recurrent.input_size, new_size, batch_first=True)
        expanded.to(recurrent.weight_ih_l0)
        for name, old_parameter in recurrent.named_parameters():
            new_parameter = getattr(expanded, name)
            gate_count = old_parameter.shape[0] // old_size
            old_gates = old_parameter.view(gate_count, old_size, *old_parameter.shape[1:])
            new_gates = new_parameter.view(gate_count, new_size, *new_parameter.shape[1:])
            if name == "weight_hh_l0":
                new_gates[:, :old_size, :old_size].copy_(old_gates)
            else:
                new_gates[:, :old_size].copy_(old_gates)

        setattr(self.learner, recurrent_name, expanded)
        self.learner.h0 = np.pad(self.learner.h0, ((0, 0), (0, hidden_mult)))
        if recurrent_name == "lstm":
            self.learner.c0 = np.pad(self.learner.c0, ((0, 0), (0, hidden_mult)))
        self.learner.hidden_size = self.hidden_size = new_size
        self.backbone_frozen = False
        self.frozen_hidden_size = old_size if freeze_backbone else 0

    def zero_frozen_gradients(self):
        if self.frozen_hidden_size:
            for name, parameter in self.learner.named_parameters():
                grad = parameter.grad
                if grad is not None:
                    gates = grad.view(-1, self.hidden_size, *grad.shape[1:])
                    old_rows = gates[:, : self.frozen_hidden_size]
                    if name.endswith("weight_hh_l0"):
                        old_rows[..., : self.frozen_hidden_size].zero_()
                    else:
                        old_rows.zero_()

    def add_task_head(self, task_id, previous=False):
        if str(task_id) in self.heads:
            return
        if previous:
            head = deepcopy(self.heads[list(self.heads)[-1]])
        else:
            head = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.heads[str(task_id)] = head
        self.set_trainable_task(task_id)

    def freeze_backbone(self):
        self.backbone_frozen = True
        for param in self.learner.parameters():
            param.requires_grad_(False)

    def set_trainable_task(self, task_id):
        for param in self.learner.parameters():
            param.requires_grad_(not self.backbone_frozen)
        for key, head in self.heads.items():
            for param in head.parameters():
                param.requires_grad_(key == str(task_id))
