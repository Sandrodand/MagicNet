import torch
from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

# importing gru math computation block
from torch._VF import gru as _VF_gru
from torch._VF import lstm as _VF_lstm
from models.magic.activations import (
    Binarizer,
    Ternarizer,
    Thresholder,
    inverseCappedSigmoid,
    inverseSigmoid,
    CAP_VALUE,
)

DEFAULT_THRESHOLD = 5e-3


def GRUBlockMath(
    input,
    hn,
    weight_thresholded_ih,
    weight_thresholded_hh,
    bias_ih_l0,
    bias_hh_l0,
    batch_size=None,
    bias=True,
    num_layers=1,
    dropout=0.0,
    training=False,
    bidirectional=False,
    batch_first=False,
):
    # print(weight_thresholded_ih)
    tensors = [weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0, bias_hh_l0]
    # print(tensors)
    # batch_size = torch.tensor(batch_size)
    batch_size = None
    if batch_size == None:
        output, new_hn = _VF_gru(
            input,
            hn,
            tensors,
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
            batch_first,
        )
    else:
        output, new_hn = _VF_gru(
            input,
            batch_size,
            hn,
            tensors,
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
        )
    # print(output)
    return output, new_hn


def LSTMBlockMath(
    input,
    hn,
    weight_thresholded_ih,
    weight_thresholded_hh,
    bias_ih_l0,
    bias_hh_l0,
    bias=True,
    num_layers=1,
    dropout=0.0,
    training=False,
    bidirectional=False,
    batch_first=False,
):
    tensors = [weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0, bias_hh_l0]
    batch_size = None
    if batch_size == None:
        results = _VF_lstm(
            input,
            hn,
            tensors,
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
            batch_first,
        )
    else:
        results = _VF_lstm(
            input,
            batch_size,
            hn,
            tensors,
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
        )
    output = results[0]
    return output


class ElementWiseGRU(nn.Module):
    """Modified GRU layer."""

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
        batch_first=True,
        mask_init="1s",
        mask_scale=2e-2,
        threshold_fn="sigmoid",
        threshold=5e-3,
        seq_len=10,
        GRU_mask_weights=[],
        GRU_weights=None,
        cap_sigmoid=True,
    ):
        super(ElementWiseGRU, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = round(hidden_size)
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.bias = bias
        self.dropout = dropout
        self.training = training
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        self.threshold = threshold
        self.seq_len = (seq_len,)
        self.GRU_mask_weights = GRU_mask_weights
        self.cap_sigmoid = cap_sigmoid
        if cap_sigmoid:
            self.inverse_thresholder = inverseCappedSigmoid
        else:
            self.inverse_thresholder = inverseSigmoid

        self.h0 = np.zeros((1, self.hidden_size))

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            "threshold_fn": threshold_fn,
            "threshold": threshold,
        }

        if GRU_weights is not None:
            self.weight_ih = Parameter(GRU_weights.weight_ih_l0)
            self.weight_hh = Parameter(GRU_weights.weight_hh_l0)
            self.bias_ih_l0 = Parameter(GRU_weights.bias_ih_l0)
            self.bias_hh_l0 = Parameter(GRU_weights.bias_hh_l0)
        else:
            self.weight_ih = Parameter(
                torch.Tensor(3 * hidden_size, input_size).uniform_(
                    -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
                )
            )
            self.weight_hh = Parameter(
                torch.Tensor(3 * hidden_size, hidden_size).uniform_(
                    -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
                )
            )

            self.bias_ih_l0 = Parameter(
                torch.Tensor(3 * hidden_size).uniform_(
                    -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
                )
            )
            self.bias_hh_l0 = Parameter(
                torch.Tensor(3 * hidden_size).uniform_(
                    -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
                )
            )

        if threshold_fn == "binarizer":
            self.threshold_fn = Binarizer()
        elif threshold_fn == "ternarizer":
            self.threshold_fn = Ternarizer()
        else:
            self.threshold_fn = Thresholder(
                function=threshold_fn, cap_sigmoid=self.cap_sigmoid
            )
            if not self.cap_sigmoid:
                self.mask_scale = 1
            else:
                self.mask_scale = CAP_VALUE

        self.initialize_piggymask(mask_init, self.mask_scale)

    def forward(self, input):
        self.hn = self._build_initial_state(input, self.h0)

        # Apply threshold function to real valued masks
        mask_thresholded_ih = self.threshold_fn(self.mask_real_weight_ih)
        mask_thresholded_hh = self.threshold_fn(self.mask_real_weight_hh)
        mask_thresholded_bias_ih_l0 = self.threshold_fn(self.mask_real_bias_ih_l0)
        mask_thresholded_bias_hh_l0 = self.threshold_fn(self.mask_real_bias_hh_l0)

        # Mask weights with above mask.
        weight_thresholded_ih = mask_thresholded_ih * self.weight_ih
        weight_thresholded_hh = mask_thresholded_hh * self.weight_hh
        weight_thresholded_bias_ih = mask_thresholded_bias_ih_l0 * self.bias_ih_l0
        weight_thresholded_bias_hh = mask_thresholded_bias_hh_l0 * self.bias_hh_l0

        out, _ = GRUBlockMath(
            input,
            self.hn,
            weight_thresholded_ih,
            weight_thresholded_hh,
            weight_thresholded_bias_ih,
            weight_thresholded_bias_hh,
            self.batch_size,
            self.bias,
            self.num_layers,
            self.dropout,
            self.training,
            self.bidirectional,
            self.batch_first,
        )
        # Get output using modified weight.

        return out

    def reinit_piggymask(self, mask_init, mask_scale, freeze_masks=None, masks=None):
        if masks is not None:
            self.mask_real_weight_ih = Parameter(
                self.inverse_thresholder(masks["mask_real_weight_ih"])
            )
            self.mask_real_weight_hh = Parameter(
                self.inverse_thresholder(masks["mask_real_weight_hh"])
            )
            self.mask_real_bias_ih_l0 = Parameter(
                self.inverse_thresholder(masks["mask_real_bias_ih_l0"])
            )
            self.mask_real_bias_hh_l0 = Parameter(
                self.inverse_thresholder(masks["mask_real_bias_hh_l0"])
            )
        else:
            self.initialize_piggymask(mask_init, mask_scale, freeze_masks=freeze_masks)

    def create_freezemask(self):
        """
        Create initial freeze masks.
        """
        freeze_masks = {}
        freeze_masks["weight_ih"] = torch.zeros_like(self.weight_ih)
        freeze_masks["weight_hh"] = torch.zeros_like(self.weight_hh)
        freeze_masks["bias_ih_l0"] = torch.zeros_like(self.bias_ih_l0)
        freeze_masks["bias_hh_l0"] = torch.zeros_like(self.bias_hh_l0)
        return freeze_masks

    def expand_hidden(self, addendum):
        """
        Parameters are expanded by a factor of "addendum". New weights are initialized normally while
        previously learned ones are kept and frozen.
        Rispettando la struttura a 3 porte (Reset, Update, New) della GRU.
        """
        old_hidden_size = self.hidden_size
        self.hidden_size = round(self.hidden_size + addendum)
        freeze_masks = {}

        # 1. Creiamo i nuovi tensori
        temp_weight_ih = torch.Tensor(3 * self.hidden_size, self.input_size).uniform_(
            -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
        )
        temp_weight_hh = torch.Tensor(3 * self.hidden_size, self.hidden_size).uniform_(
            -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
        )
        temp_bias_ih_l0 = torch.Tensor(3 * self.hidden_size).uniform_(
            -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
        )
        temp_bias_hh_l0 = torch.Tensor(3 * self.hidden_size).uniform_(
            -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
        )

        freeze_masks["weight_ih"] = torch.zeros_like(temp_weight_ih)
        freeze_masks["weight_hh"] = torch.zeros_like(temp_weight_hh)
        freeze_masks["bias_ih_l0"] = torch.zeros_like(temp_bias_ih_l0)
        freeze_masks["bias_hh_l0"] = torch.zeros_like(temp_bias_hh_l0)

        for i in range(3):
            old_start = i * old_hidden_size
            old_end = (i + 1) * old_hidden_size
            new_start = i * self.hidden_size
            new_end = new_start + old_hidden_size

            temp_weight_ih[new_start:new_end, :].copy_(self.weight_ih[old_start:old_end, :])
            freeze_masks["weight_ih"][new_start:new_end, :].fill_(1)

            temp_weight_hh[new_start:new_end, :old_hidden_size].copy_(self.weight_hh[old_start:old_end, :])
            freeze_masks["weight_hh"][new_start:new_end, :old_hidden_size].fill_(1)

            temp_bias_ih_l0[new_start:new_end].copy_(self.bias_ih_l0[old_start:old_end])
            freeze_masks["bias_ih_l0"][new_start:new_end].fill_(1)

            temp_bias_hh_l0[new_start:new_end].copy_(self.bias_hh_l0[old_start:old_end])
            freeze_masks["bias_hh_l0"][new_start:new_end].fill_(1)

        self.weight_ih = Parameter(temp_weight_ih)
        self.weight_hh = Parameter(temp_weight_hh)
        self.bias_ih_l0 = Parameter(temp_bias_ih_l0)
        self.bias_hh_l0 = Parameter(temp_bias_hh_l0)

        self.h0 = np.zeros((1, self.hidden_size))
        self.adjust_piggymask(mask_scale=self.mask_scale, freeze_masks=freeze_masks)

        return freeze_masks

    def adjust_piggymask(self, mask_scale, freeze_masks):
        old_hidden_size = self.mask_real_weight_ih.size(0) // 3

        temp_mask_real_weight_ih = self.weight_ih.data.new(self.weight_ih.size()).fill_(mask_scale)
        temp_mask_real_weight_hh = self.weight_hh.data.new(self.weight_hh.size()).fill_(mask_scale)
        temp_mask_real_bias_ih_l0 = self.weight_ih.data.new(self.bias_ih_l0.size()).fill_(mask_scale)
        temp_mask_real_bias_hh_l0 = self.weight_hh.data.new(self.bias_hh_l0.size()).fill_(mask_scale)

        # Trasloco a blocchi anche per le maschere continue!
        for i in range(3):
            old_start = i * old_hidden_size
            old_end = (i + 1) * old_hidden_size
            new_start = i * self.hidden_size
            new_end = new_start + old_hidden_size

            temp_mask_real_weight_ih[new_start:new_end, :].copy_(self.mask_real_weight_ih[old_start:old_end, :])
            temp_mask_real_weight_hh[new_start:new_end, :old_hidden_size].copy_(
                self.mask_real_weight_hh[old_start:old_end, :])
            temp_mask_real_bias_ih_l0[new_start:new_end].copy_(self.mask_real_bias_ih_l0[old_start:old_end])
            temp_mask_real_bias_hh_l0[new_start:new_end].copy_(self.mask_real_bias_hh_l0[old_start:old_end])

        self.mask_real_weight_ih = Parameter(temp_mask_real_weight_ih)
        self.mask_real_weight_hh = Parameter(temp_mask_real_weight_hh)
        self.mask_real_bias_ih_l0 = Parameter(temp_mask_real_bias_ih_l0)
        self.mask_real_bias_hh_l0 = Parameter(temp_mask_real_bias_hh_l0)

    def initialize_piggymask(self, mask_init, mask_scale, freeze_masks=None):
        temp_mask_real_weight_ih = self.weight_ih.data.new(self.weight_ih.size())
        temp_mask_real_weight_hh = self.weight_hh.data.new(self.weight_hh.size())
        temp_mask_real_bias_ih_l0 = self.weight_ih.data.new(self.bias_ih_l0.size())
        temp_mask_real_bias_hh_l0 = self.weight_hh.data.new(self.bias_hh_l0.size())

        if mask_init == "1s":
            temp_mask_real_weight_ih.fill_(mask_scale)
            temp_mask_real_weight_hh.fill_(mask_scale)
            temp_mask_real_bias_ih_l0.fill_(mask_scale)
            temp_mask_real_bias_hh_l0.fill_(mask_scale)
        elif mask_init == "uniform":
            temp_mask_real_weight_ih.uniform_(-1 * mask_scale, mask_scale)
            temp_mask_real_weight_hh.uniform_(-1 * mask_scale, mask_scale)
            temp_mask_real_bias_ih_l0.uniform_(-1 * mask_scale, mask_scale)
            temp_mask_real_bias_hh_l0.uniform_(-1 * mask_scale, mask_scale)

        if freeze_masks is not None:
            temp_mask_real_weight_ih[
                freeze_masks["weight_ih"].eq(1)
            ] = temp_mask_real_weight_ih[freeze_masks["weight_ih"].eq(1)].uniform_(
                -1 * mask_scale, mask_scale
            )
            temp_mask_real_weight_hh[
                freeze_masks["weight_hh"].eq(1)
            ] = temp_mask_real_weight_hh[freeze_masks["weight_hh"].eq(1)].uniform_(
                -1 * mask_scale, mask_scale
            )
            temp_mask_real_bias_ih_l0[
                freeze_masks["bias_ih_l0"].eq(1)
            ] = temp_mask_real_bias_ih_l0[freeze_masks["bias_ih_l0"].eq(1)].uniform_(
                -1 * mask_scale, mask_scale
            )
            temp_mask_real_bias_hh_l0[
                freeze_masks["bias_hh_l0"].eq(1)
            ] = temp_mask_real_bias_hh_l0[freeze_masks["bias_hh_l0"].eq(1)].uniform_(
                -1 * mask_scale, mask_scale
            )

        self.mask_real_weight_ih = Parameter(temp_mask_real_weight_ih)
        self.mask_real_weight_hh = Parameter(temp_mask_real_weight_hh)
        self.mask_real_bias_ih_l0 = Parameter(temp_mask_real_bias_ih_l0)
        self.mask_real_bias_hh_l0 = Parameter(temp_mask_real_bias_hh_l0)

    def _build_initial_state(self, x, state):
        s = torch.from_numpy(np.tile(state, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)


class ElementWiseLinear(nn.Module):
    """Modified linear layer."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mask_init="1s",
        mask_scale=2e-2,
        threshold_fn="sigmoid",
        threshold=5e-3,
        Linear_mask_weights=[],
        linear_weights=None,
        cap_sigmoid=True,
    ):
        super(ElementWiseLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.Linear_mask_weights = Linear_mask_weights
        self.cap_sigmoid = cap_sigmoid
        if self.cap_sigmoid:
            self.inverse_thresholder = inverseCappedSigmoid
        else:
            self.inverse_thresholder = inverseSigmoid
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            "threshold_fn": threshold_fn,
            "threshold": threshold,
        }

        if linear_weights is not None:
            self.weight = Parameter(linear_weights.weight)
            self.bias = Parameter(linear_weights.bias)
        else:
            self.weight = Parameter(
                torch.Tensor(out_features, in_features).uniform_(
                    -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
                )
            )
            if bias:
                self.bias = Parameter(
                    torch.Tensor(out_features).uniform_(
                        -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
                    )
                )
            else:
                self.register_parameter("bias", None)

        # Initialize the thresholder.
        if threshold_fn == "binarizer":
            self.threshold_fn = Binarizer()
        elif threshold_fn == "ternarizer":
            self.threshold_fn = Ternarizer()
        else:
            self.threshold_fn = Thresholder(
                function=threshold_fn, cap_sigmoid=self.cap_sigmoid
            )
            if self.cap_sigmoid:
                self.mask_scale = CAP_VALUE
            else:
                self.mask_scale = 1
        self.initialize_piggymask(mask_init, self.mask_scale, Linear_mask_weights)

    def reinit_piggymask(self, mask_init, mask_scale, freeze_masks=None, masks=None):
        if masks is not None:
            self.mask_real_weight = Parameter(
                self.inverse_thresholder(masks["mask_real_weight"])
            )
        else:
            self.initialize_piggymask(mask_init, mask_scale, freeze_mask=freeze_masks)

    def reinit_linear_bias(self):
        if self.bias is not None:
            self.bias = Parameter(
                torch.Tensor(self.out_features).uniform_(
                    -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
                )
            )
        else:
            self.register_parameter("bias", None)

    def create_freezemask(self):
        freeze_masks = {}
        freeze_masks["weight"] = torch.zeros_like(self.weight)
        return freeze_masks

    def initialize_piggymask(
        self, mask_init, mask_scale, Linear_mask_weights=[], freeze_mask=None
    ):
        temp_mask_real_weight = self.weight.data.new(self.weight.size())
        if mask_init == "1s":
            temp_mask_real_weight.fill_(mask_scale)
        elif mask_init == "uniform":
            temp_mask_real_weight.uniform_(-1 * mask_scale, mask_scale)

        if freeze_mask is not None:
            temp_mask_real_weight[freeze_mask["weight"].eq(1)] = temp_mask_real_weight[
                freeze_mask["weight"].eq(1)
            ].uniform_(-1 * mask_scale, mask_scale)

        if Linear_mask_weights != []:
            self.mask_real_weight = Parameter(Linear_mask_weights)
        else:
            self.mask_real_weight = Parameter(temp_mask_real_weight)

    def expand_hidden(self, multiplier):
        freeze_mask = {}
        self.in_features = round(self.in_features + multiplier)
        temp_weight = torch.Tensor(self.out_features, self.in_features).uniform_(
            -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
        )
        freeze_mask["weight"] = torch.zeros_like(temp_weight)
        temp_weight[: self.weight.size(0), : self.weight.size(1)].copy_(self.weight)
        freeze_mask["weight"][: self.weight.size(0), : self.weight.size(1)].fill_(1)
        self.weight = Parameter(temp_weight)
        if self.bias is not None:
            self.bias = Parameter(
                torch.Tensor(self.out_features).uniform_(
                    -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
                )
            )
        else:
            self.register_parameter("bias", None)
        self.adjust_piggymask(self.mask_scale, freeze_masks=freeze_mask)
        return freeze_mask

    def adjust_piggymask(self, mask_scale, freeze_masks):
        temp_mask_real_weight = self.weight.data.new(self.weight.size())

        temp_mask_real_weight.fill_(mask_scale)

        temp_mask_real_weight[
            : self.mask_real_weight.size(0), : self.mask_real_weight.size(1)
        ].copy_(self.mask_real_weight)

        self.mask_real_weight = Parameter(temp_mask_real_weight)

    def forward(self, input):
        # Get transformed mask from real-valued mask.
        mask_thresholded = self.threshold_fn(self.mask_real_weight)
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        a = F.linear(input[:, -1, :], weight_thresholded, self.bias)
        # Get output using modified weight.
        return a
