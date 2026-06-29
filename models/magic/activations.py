import torch
from torch import nn as nn, Tensor

CAP_VALUE = 13.8


class BinarizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        # Straight-through estimator: genera 1 se > threshold, altrimenti 0
        outputs = torch.zeros_like(inputs)
        outputs[inputs >= threshold] = 1.0
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Binarizer(nn.Module):
    """Binarizer {0, 1} a real-valued tensor."""

    def __init__(self, threshold=5e-3):
        super(Binarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        return BinarizerFunction.apply(inputs, self.threshold)


class TernarizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < -threshold] = -1
        outputs[inputs > threshold] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients through unchanged
        return grad_output, None  # None for the threshold


class Ternarizer(nn.Module):
    """Ternarizes {-1, 0, 1} a real-valued tensor."""

    def __init__(self, threshold=5e-3):
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        return TernarizerFunction.apply(inputs, self.threshold)


def linear(input):
    return input


class Thresholder(nn.Module):
    """Ternarizes {-1, 0, 1} a real-valued tensor."""

    def __init__(self, function="ternarizer", cap_sigmoid=True):
        super(Thresholder, self).__init__()
        self.threshold = 5e-3
        if function == "linear":
            self.threshold_fn = linear
        elif function == "sigmoid":
            if not cap_sigmoid:
                self.threshold_fn = nn.Sigmoid()
            else:
                self.threshold_fn = CappedSigmoid()
        elif function == "relu":
            self.threshold_fn = nn.LeakyReLU(negative_slope=0.05)
        elif function == "tanh":
            self.threshold_fn = nn.Tanh()

    def forward(self, inputs):
        return self.threshold_fn(inputs)


class CappedSigmoid(nn.Module):
    def __init__(self, cap_value=CAP_VALUE):
        super(CappedSigmoid, self).__init__()
        self.cap_value = float(cap_value)

    def forward(self, x: Tensor) -> Tensor:
        cap_tensor = torch.tensor(self.cap_value, dtype=x.dtype, device=x.device)
        ones = torch.ones_like(x)
        return torch.where(x.ge(cap_tensor), ones, torch.sigmoid(x))


def inverseCappedSigmoid(tensor):
    """
    Computes the inverse sigmoid (logit) of all values in a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor with values between 0 and 1 (exclusive).

    Returns:
        torch.Tensor: A tensor with the inverse sigmoid applied element-wise.
    """
    # Ensure values are within valid range for logit calculation
    epsilon = 1e-6  # To prevent division by zero or log of zero
    tensor = torch.clamp(tensor, epsilon, 1 - epsilon)
    logit = torch.log(tensor / (1 - tensor))
    return torch.where(
        logit.ge(CAP_VALUE),
        torch.tensor(CAP_VALUE, dtype=logit.dtype, device=logit.device),
        logit,
    )


def inverseSigmoid(tensor):
    """
    Computes the inverse sigmoid (logit) of all values in a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor with values between 0 and 1 (exclusive).

    Returns:
        torch.Tensor: A tensor with the inverse sigmoid applied element-wise.
    """
    # Ensure values are within valid range for logit calculation
    epsilon = 1e-6  # To prevent division by zero or log of zero
    tensor = torch.clamp(tensor, epsilon, 1 - epsilon)
    return torch.log(tensor / (1 - tensor))
