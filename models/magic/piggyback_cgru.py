# this file is for Main network that contains piggyback layers with masks.
import copy

import torch
import torch.nn as nn
import models.magic.piggyback_layers as nl
from models.magic.activations import CappedSigmoid


class PiggyBackGRU(nn.Module):
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
        mask_init="1s",
        mask_scale=2e-2,
        threshold_fn="sigmoid",
        threshold=None,
        seq_len=5,
        mask_weights=[],
        cGRU_weights=None,
        cap_sigmoid=True,
        multi_head=True,
        initial_task_id=1
    ):
        super(PiggyBackGRU, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.seq_len = seq_len
        self.mask_weights = mask_weights
        self.cap_sigmoid = cap_sigmoid
        self.multi_head = multi_head

        if mask_weights != []:
            self.GRU_mask_weights = mask_weights[0:4]
            self.Linear_mask_weights = mask_weights[-1]
        else:
            self.GRU_mask_weights = []
            self.Linear_mask_weights = []
        # define nn network here

        if cGRU_weights is not None:
            GRU_weights = cGRU_weights.columns.columns[0].gru
            linear_weights = cGRU_weights.columns.columns[0].linear
        else:
            GRU_weights = None
            linear_weights = None

        gru_layer = nl.ElementWiseGRU(input_size=input_size, device=device, num_layers=num_layers, hidden_size=hidden_size,
                              bias=bias, dropout=dropout, training=training, bidirectional=bidirectional,
                              mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,
                              threshold=threshold, seq_len=self.seq_len, GRU_mask_weights=self.GRU_mask_weights,
                              GRU_weights=GRU_weights, cap_sigmoid=cap_sigmoid)

        if not self.multi_head:
            self.classifier = nn.Sequential(
                gru_layer,
                nl.ElementWiseLinear(
                    in_features=hidden_size,
                    out_features=output_size,
                    mask_init=mask_init,
                    mask_scale=mask_scale,
                    threshold_fn=threshold_fn,
                    threshold=threshold,
                    Linear_mask_weights=self.Linear_mask_weights,
                    linear_weights=linear_weights,
                    cap_sigmoid=cap_sigmoid,
                )
            )
        else:
            self.classifier = nn.Sequential(gru_layer)
            self.heads = nn.ModuleDict()
            self.heads[str(initial_task_id)] = linear_weights
            self.heads.to(self.device)

        self.classifier.to(self.device)

    def forward(self, input, task_id=None):
        if not self.multi_head:
            out = self.classifier(input)
        else:
            gru_out = self.classifier[0](input)
            if task_id is None or str(task_id) not in self.heads:
                task_id = list(self.heads.keys())[-1]
            head = self.heads[str(task_id)]
            expected_features = head.in_features
            out = head(gru_out[:, -1, :expected_features])
        return out

    def add_task_head(self, task_id, previous=False):
        if self.multi_head and str(task_id) not in self.heads:
            if not previous:
                self.heads[str(task_id)] = nn.Linear(self.hidden_size, self.output_size).to(self.device)
            else:
                self.heads[str(task_id)] = copy.deepcopy(self.heads[str(task_id-1)])

    def _pad_historical_masks(self, historical_masks):
        """
        It adapts an old mask to the current model's shape. It uses -50.0 as padding to guarantee that the corresponding
         sigmoid value is 0.
        """
        # Prendiamo le maschere attuali per sapere quali sono le dimensioni "giuste" di oggi
        reference_masks = self.get_piggymasks()
        padded_masks = {}

        for name, old_tensor in historical_masks.items():
            if name not in reference_masks:
                continue

            target_shape = reference_masks[name].shape
            if old_tensor.shape == target_shape:
                padded_masks[name] = old_tensor
                continue

            new_tensor = torch.full(target_shape, -50.0, device=old_tensor.device)

            # Managing GRU layers (with 3 concatenated gates: Reset, Update, New)
            if 'weight_ih' in name or 'weight_hh' in name or 'bias_ih' in name or 'bias_hh' in name:
                old_hidden = old_tensor.shape[0] // 3
                new_hidden = target_shape[0] // 3

                if len(target_shape) == 1:  # Maschere 1D o Bias
                    for i in range(3):
                        new_tensor[i * new_hidden: i * new_hidden + old_hidden] = old_tensor[
                                                                                  i * old_hidden: (i + 1) * old_hidden]
                elif len(target_shape) == 2:  # Maschere 2D (Weights)
                    old_in = old_tensor.shape[1]
                    for i in range(3):
                        new_tensor[i * new_hidden: i * new_hidden + old_hidden, :old_in] = old_tensor[i * old_hidden: (
                                                                                                                                  i + 1) * old_hidden,
                                                                                           :]
            else:
                # Layer standard non-GRU (es. teste lineari)
                if len(target_shape) == 1:
                    new_tensor[:old_tensor.shape[0]] = old_tensor
                elif len(target_shape) == 2:
                    new_tensor[:old_tensor.shape[0], :old_tensor.shape[1]] = old_tensor

            padded_masks[name] = new_tensor

        return padded_masks

    def reinit_piggymask(self, mask_init, freeze_masks=None, masks=None):
        if masks is not None:
            masks = self._pad_historical_masks(masks)
        for module in self.classifier:
            module.reinit_piggymask(
                mask_init, self.mask_scale, freeze_masks=freeze_masks, masks=masks
            )

    def reinit_linear_bias(self):
        if not self.multi_head:
            self.classifier[1].reinit_linear_bias()

    def expand_hidden(self, multiplier):
        self.hidden_size = round(self.hidden_size + multiplier)
        freeze_masks = {}
        for module in self.classifier:
            freeze_masks.update(module.expand_hidden(multiplier))
        if self.multi_head:
            current_task_id = list(self.heads.keys())[-1]
            self.heads[current_task_id] = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.classifier.to(self.device)
        return freeze_masks

    def create_freezemask(self):
        freeze_mask = {}
        for module in self.classifier:
            freeze_mask.update(module.create_freezemask())
        return freeze_mask

    def get_piggymasks(self):
        if self.cap_sigmoid:
            sigmoid = CappedSigmoid()
        else:
            sigmoid = nn.Sigmoid()
        piggymasks = {}

        piggymasks["mask_real_weight_ih"] = sigmoid(
            self.classifier[0].mask_real_weight_ih
        )
        piggymasks["mask_real_weight_hh"] = sigmoid(
            self.classifier[0].mask_real_weight_hh
        )
        piggymasks["mask_real_bias_ih_l0"] = sigmoid(
            self.classifier[0].mask_real_bias_ih_l0
        )
        piggymasks["mask_real_bias_hh_l0"] = sigmoid(
            self.classifier[0].mask_real_bias_hh_l0
        )
        if not self.multi_head:
            piggymasks["mask_real_weight"] = sigmoid(self.classifier[1].mask_real_weight)
        return piggymasks
