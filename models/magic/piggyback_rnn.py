"""Shared masked recurrent model lifecycle."""
import copy
from math import log

import torch
import torch.nn as nn
import models.magic.piggyback_layers as nl


class PiggyBackRNN(nn.Module):
    """
    Common masks, heads and structural expansion for masked recurrent models.
    Subclasses select their recurrent implementation through class attributes.
    """

    recurrent_name = None
    recurrent_layer = None

    def __init__(
        self,
        input_size=4,
        device=torch.device("cpu"),
        num_layers=1,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        bias=True,
        dropout=0.0,
        training=False,
        bidirectional=False,
        frozen_mask_param=2e-2,
        threshold_fn="sigmoid",
        threshold=None,
        seq_len=5,
        mask_weights=None,
        multi_head=True,
        base_learner_weights=None,
    ):
        """
        Parameters
        ----------
        input_size : int, default: 4
            Dimensionality of input features.
        device : torch.device, default: "cpu"
            Computation device.
        num_layers : int, default: 1
            Number of recurrent layers.
        hidden_size : int, default: 50
            Number of recurrent hidden units.
        output_size : int, default: 2
            Number of output classes.
        batch_size : int, default: 128
            Training mini-batch size.
        bias : bool, default: True
            Whether to use recurrent bias weights.
        dropout : float, default: 0.0
            Dropout probability.
        training : bool, default: False
            Sets the training state.
        bidirectional : bool, default: False
            If True, uses a bidirectional recurrent layer.
        frozen_mask_param : float, default: 2e-2
            Initialization parameter for the underlying mask values.
        threshold_fn : str, default: "sigmoid"
            Function used to threshold real-valued masks.
        threshold : float, default: None
            Specific threshold cutoff value (if using binarization).
        seq_len : int, default: 5
            Length of the temporal sequence.
        mask_weights : list, default: None
            Pre-initialized mask weights to load.
        multi_head : bool, default: True
            If True, instantiates a separate output head for each task.
        base_learner_weights : object, default: None
            Shared cPNN base learner or recurrent column. Its dimensions and
            parameters are reused through independent copies.
        """
        super().__init__()
        if self.recurrent_name is None or self.recurrent_layer is None:
            raise TypeError("Instantiate a concrete PiggyBackRNN subclass")

        # PARAMETERS
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.threshold_fn = threshold_fn
        self.seq_len = seq_len
        self.mask_weights = (
            [] if mask_weights is None else
            [torch.as_tensor(mask).detach().clone() for mask in mask_weights]
        )
        self.multi_head = multi_head

        if frozen_mask_param is None:
            if self.threshold_fn == "binarizer":
                frozen_mask_param = 1e-2
            else:
                frozen_mask_param = 2e-2
        self.frozen_mask_param = frozen_mask_param

        parameter_count = len(self.recurrent_layer.mask_parameter_names)
        self.recurrent_mask_weights = self.mask_weights[:parameter_count]
        self.linear_mask_weights = (
            self.mask_weights[parameter_count]
            if len(self.mask_weights) > parameter_count else None
        )

        source = base_learner_weights
        if source is not None:
            if hasattr(source, "columns"):
                source = source.columns.columns[0]
            if not hasattr(source, self.recurrent_name):
                raise ValueError("Base learner does not match the recurrent architecture")
            recurrent_weights = getattr(source, self.recurrent_name)
            linear_weights = source.linear
            input_size = self.input_size = recurrent_weights.input_size
            hidden_size = self.hidden_size = recurrent_weights.hidden_size
            output_size = self.output_size = linear_weights.out_features
        else:
            recurrent_weights = None
            linear_weights = None

        recurrent_layer = self.recurrent_layer(
            input_size=input_size,
            device=device,
            num_layers=num_layers,
            hidden_size=hidden_size,
            bias=bias,
            dropout=dropout,
            training=training,
            bidirectional=bidirectional,
            frozen_mask_param=frozen_mask_param,
            threshold_fn=threshold_fn,
            threshold=threshold,
            seq_len=self.seq_len,
            recurrent_mask_weights=self.recurrent_mask_weights,
            recurrent_weights=recurrent_weights,
        )

        if not self.multi_head:
            self.classifier = nn.Sequential(
                recurrent_layer,
                nl.ElementWiseLinear(
                    in_features=hidden_size,
                    out_features=output_size,
                    frozen_mask_param=frozen_mask_param,
                    threshold_fn=threshold_fn,
                    threshold=threshold,
                    linear_mask_weights=self.linear_mask_weights,
                    linear_weights=linear_weights,
                ),
            )
        else:
            self.classifier = nn.Sequential(recurrent_layer)
            self.heads = nn.ModuleDict()
            self.heads["1"] = (
                copy.deepcopy(linear_weights)
                if linear_weights is not None
                else nn.Linear(hidden_size, output_size)
            )
            self.heads.to(self.device)

        self.classifier.to(self.device)

    def forward(self, input, task_id=None):
        """
        Performs a forward pass applying the current masks to the frozen weights.

        Parameters
        ----------
        input : torch.Tensor
            Input feature tensor.
        task_id : int, optional
            The specific task ID to select the corresponding classification head
            (if multi_head is True). Defaults to the latest task.

        Returns
        -------
        torch.Tensor
            The output predictions.
        """
        if not self.multi_head:
            out = self.classifier(input)
        else:
            recurrent_out = self.classifier[0](input)
            if task_id is None or str(task_id) not in self.heads:
                task_id = list(self.heads.keys())[-1]
            head = self.heads[str(task_id)]
            expected_features = head.in_features
            out = head(recurrent_out[:, -1, :expected_features])
        return out

    def add_task_head(self, task_id, previous=False):
        """Add a task head when enabled, optionally copying the latest head."""
        if self.multi_head and str(task_id) not in self.heads:
            if not previous:
                self.heads[str(task_id)] = nn.Linear(
                    self.hidden_size, self.output_size
                ).to(self.device)
            else:
                last_head_key = list(self.heads.keys())[-1]
                self.heads[str(task_id)] = copy.deepcopy(self.heads[last_head_key])

    def _pad_historical_masks(self, historical_masks):
        """
        It adapts effective historical masks to the current model's shape.
        New connections are padded with zero before conversion back to logits.
        """
        # Use the current masks to determine the target shapes.
        reference_masks = self.get_piggymasks()
        gate_count = self.classifier[0].gate_count
        padded_masks = {}

        for name, old_tensor in historical_masks.items():
            if name not in reference_masks:
                continue

            target_shape = reference_masks[name].shape
            if old_tensor.shape == target_shape:
                padded_masks[name] = old_tensor
                continue

            new_tensor = old_tensor.new_zeros(target_shape)

            # Preserve each gate block independently of the recurrent architecture.
            if (
                "weight_ih" in name
                or "weight_hh" in name
                or "bias_ih" in name
                or "bias_hh" in name
            ):
                old_hidden = old_tensor.shape[0] // gate_count
                new_hidden = target_shape[0] // gate_count

                if len(target_shape) == 1:  # Bias mask
                    for i in range(gate_count):
                        new_tensor[
                            i * new_hidden : i * new_hidden + old_hidden
                        ] = old_tensor[i * old_hidden : (i + 1) * old_hidden]
                elif len(target_shape) == 2:  # Weight mask
                    old_in = old_tensor.shape[1]
                    for i in range(gate_count):
                        new_tensor[
                            i * new_hidden : i * new_hidden + old_hidden, :old_in
                        ] = old_tensor[i * old_hidden : (i + 1) * old_hidden, :]
            else:
                # Non-recurrent parameters, such as the linear head.
                if len(target_shape) == 1:
                    new_tensor[: old_tensor.shape[0]] = old_tensor
                elif len(target_shape) == 2:
                    new_tensor[
                        : old_tensor.shape[0], : old_tensor.shape[1]
                    ] = old_tensor

            padded_masks[name] = new_tensor

        return padded_masks

    def reinit_piggymask(self, freeze_masks=None, mask_params=None):
        """Initialize task masks or restore effective historical masks at the current width."""
        if mask_params is not None:
            mask_params = self._pad_historical_masks(mask_params)
        for module in self.classifier:
            module.reinit_piggymask(
                self.frozen_mask_param,
                freeze_masks=freeze_masks,
                mask_params=mask_params,
            )

    def reinit_linear_bias(self):
        """Reinitialize the shared classifier bias in single-head mode."""
        if not self.multi_head:
            self.classifier[1].reinit_linear_bias()

    def reinit_newly_frozen_masks(self, previous_freeze_masks):
        """Make newly frozen sigmoid connections adaptable in Mask-Last.

        Previously trainable weights have a zero in the old freeze mask. Their
        transparent masks start at 0.9 instead of remaining at the sigmoid cap.
        Existing learnable masks keep their logits. Binary masks already use
        an STE and retain their previous initialization.
        """
        if self.threshold_fn != "sigmoid":
            return
        mask_logit = log(0.9 / (1 - 0.9))
        with torch.no_grad():
            for module in self.classifier:
                for name, param in module.named_parameters():
                    if name.startswith("mask_real_"):
                        key = name[len("mask_real_") :]
                        newly_frozen = previous_freeze_masks[key].eq(0).to(param.device)
                        param.masked_fill_(newly_frozen, mask_logit)

    def expand_hidden(self, neurons_to_add):
        """
        Expands the hidden state capacity of the recurrent layer by allocating new neurons,
        while retaining and freezing previously learned weights.

        Parameters
        ----------
        neurons_to_add : int
            The number of new hidden units to add to the layer.

        Returns
        -------
        dict
            Freeze masks with ones for existing parameters and zeros for new
            parameters, which remain trainable.
        """
        self.hidden_size = round(self.hidden_size + neurons_to_add)
        freeze_masks = {}
        for module in self.classifier:
            freeze_masks.update(module.expand_hidden(neurons_to_add))
        if self.multi_head:
            current_task_id = list(self.heads.keys())[-1]
            self.heads[current_task_id] = nn.Linear(
                self.hidden_size, self.output_size
            ).to(self.device)
        self.classifier.to(self.device)
        return freeze_masks

    def create_freezemask(self):
        """Return all-zero freeze masks for the masked layers."""
        freeze_mask = {}
        for module in self.classifier:
            freeze_mask.update(module.create_freezemask())
        return freeze_mask

    def get_piggymasks(self, threshold=True):
        """Return detached copies of the effective masks or their raw logits."""
        piggymasks = {}

        if threshold:
            piggymasks["mask_real_weight_ih"] = (
                self.classifier[0]
                .threshold_fn(self.classifier[0].mask_real_weight_ih)
                .detach()
                .clone()
            )
            piggymasks["mask_real_weight_hh"] = (
                self.classifier[0]
                .threshold_fn(self.classifier[0].mask_real_weight_hh)
                .detach()
                .clone()
            )
            piggymasks["mask_real_bias_ih_l0"] = (
                self.classifier[0]
                .threshold_fn(self.classifier[0].mask_real_bias_ih_l0)
                .detach()
                .clone()
            )
            piggymasks["mask_real_bias_hh_l0"] = (
                self.classifier[0]
                .threshold_fn(self.classifier[0].mask_real_bias_hh_l0)
                .detach()
                .clone()
            )
            if not self.multi_head:
                piggymasks["mask_real_weight"] = (
                    self.classifier[1]
                    .threshold_fn(self.classifier[1].mask_real_weight)
                    .detach()
                    .clone()
                )
        else:
            piggymasks["mask_real_weight_ih"] = (
                self.classifier[0].mask_real_weight_ih.detach().clone()
            )
            piggymasks["mask_real_weight_hh"] = (
                self.classifier[0].mask_real_weight_hh.detach().clone()
            )
            piggymasks["mask_real_bias_ih_l0"] = (
                self.classifier[0].mask_real_bias_ih_l0.detach().clone()
            )
            piggymasks["mask_real_bias_hh_l0"] = (
                self.classifier[0].mask_real_bias_hh_l0.detach().clone()
            )
            if not self.multi_head:
                piggymasks["mask_real_weight"] = (
                    self.classifier[1].mask_real_weight.detach().clone()
                )

        return piggymasks
