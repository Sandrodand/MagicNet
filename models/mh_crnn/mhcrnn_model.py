"""Unmasked recurrent base learner with an independent head for each task."""

from copy import deepcopy

import torch
from torch import nn

from models.crnn.cgru import cGRULinear
from models.crnn.clstm import cLSTMLinear


class MultiHeadCrnnModel(nn.Module):
    def __init__(
        self,
        model_class=cGRULinear,
        input_size=4,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        device="cpu",
        initial_task_id=1,
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
        self.heads = nn.ModuleDict({str(initial_task_id): self.learner.linear})
        self.learner.linear = nn.Identity()
        self.backbone_frozen = False
        self.to(self.device)
        self.set_trainable_task(initial_task_id)

    def forward(self, x, task_id=None):
        if task_id is None:
            task_id = list(self.heads)[-1]
        _, hidden = self.learner(x, None)
        return self.heads[str(task_id)](hidden[:, -1, :])

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
