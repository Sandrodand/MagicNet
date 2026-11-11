import torch
from torch import nn as nn, Tensor

CAP_VALUE = 13.8


class BinarizerFunction(torch.autograd.Function):
    """
    Custom autograd function for binarizing a tensor using a straight-through estimator
    for the backward pass.
    """

    @staticmethod
    def forward(ctx, inputs, threshold):
        """
        Applies a binarization threshold.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to binarize.
        threshold : float
            The threshold value. Values >= threshold become 1.0, otherwise 0.0.

        Returns
        -------
        torch.Tensor
            Binarized output tensor.
        """
        # The forward pass produces 1 at or above the threshold and 0 otherwise.
        outputs = torch.zeros_like(inputs)
        outputs[inputs >= threshold] = 1.0
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """Pass the gradient straight through to the mask logits."""
        return grad_output, None


class Binarizer(nn.Module):
    """Binarizer {0, 1} a real-valued tensor."""

    def __init__(self, threshold=5e-3):
        """Set the cutoff for producing an effective mask value of one."""
        super(Binarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        """Convert logits to binary mask values using a straight-through gradient."""
        return BinarizerFunction.apply(inputs, self.threshold)


class TernarizerFunction(torch.autograd.Function):
    """
    Custom autograd function for ternarizing a tensor using a straight-through estimator
    for the backward pass.
    """

    @staticmethod
    def forward(ctx, inputs, threshold):
        """
        Applies a ternarization threshold.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to ternarize.
        threshold : float
            The threshold value. Values > threshold become 1.0, values < -threshold
            become -1.0, and values in between become 0.0.

        Returns
        -------
        torch.Tensor
            Ternarized output tensor.
        """
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < -threshold] = -1
        outputs[inputs > threshold] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """Pass the gradient straight through to the mask logits."""
        return grad_output, None  # None for the threshold


class Ternarizer(nn.Module):
    """Ternarizes {-1, 0, 1} a real-valued tensor."""

    def __init__(self, threshold=5e-3):
        """Set the threshold magnitude used to form ternary mask values."""
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        """Convert logits to minus one, zero, or one with a straight-through gradient."""
        return TernarizerFunction.apply(inputs, self.threshold)


def linear(input):
    """Return the input tensor unchanged."""
    return input


class Thresholder(nn.Module):
    """
    Applies a thresholding or activation function to a real-valued tensor.

    Use 'linear', 'sigmoid' (capped), 'relu' (leaky), or 'tanh'.
    Binary and ternary masks use Binarizer and Ternarizer directly.
    """

    def __init__(self, function="ternarizer"):
        """Configure the activation for continuous mask logits."""
        super(Thresholder, self).__init__()
        self.threshold = 5e-3
        if function == "linear":
            self.threshold_fn = linear
        elif function == "sigmoid":
            self.threshold_fn = CappedSigmoid()
        elif function == "relu":
            self.threshold_fn = nn.LeakyReLU(negative_slope=0.05)
        elif function == "tanh":
            self.threshold_fn = nn.Tanh()

    def forward(self, inputs):
        """Apply the configured activation to the input logits."""
        return self.threshold_fn(inputs)


class CappedSigmoid(nn.Module):
    """
    Applies a capped sigmoid activation function.

    The function behaves like a standard sigmoid but returns 1.0 for inputs
    greater than or equal to a specified cap_value.
    """

    def __init__(self, cap_value=CAP_VALUE):
        """
        Parameters
        ----------
        cap_value : float, default: CAP_VALUE (13.8)
            The threshold above which the activation output is capped at 1.0.
        """
        super(CappedSigmoid, self).__init__()
        self.cap_value = float(cap_value)

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid and return exactly one at or above the cap."""
        cap_tensor = torch.tensor(self.cap_value, dtype=x.dtype, device=x.device)
        ones = torch.ones_like(x)
        return torch.where(x.ge(cap_tensor), ones, torch.sigmoid(x))


def inverseCappedSigmoid(tensor):
    """
    Reconstructs logits from effective historical masks, including exact zeros.

    Args:
        tensor (torch.Tensor): Effective mask values in [0, 1].

    Returns:
        torch.Tensor: Logits with -inf for zero and CAP_VALUE for one.
            These boundary values restore disabled/transparent connections
            during historical inference; they are not trainable initializers.
    """
    logit = torch.logit(tensor)
    return torch.where(
        tensor.eq(1),
        torch.tensor(CAP_VALUE, dtype=logit.dtype, device=logit.device),
        logit,
    )


def inverseSigmoid(tensor):
    """
    Legacy helper retained for loading existing pickle checkpoints.

    New layers always use inverseCappedSigmoid.

    Args:
        tensor (torch.Tensor): Input tensor with values between 0 and 1 (exclusive).

    Returns:
        torch.Tensor: A tensor with the inverse sigmoid applied element-wise.
    """
    # Ensure values are within valid range for logit calculation
    epsilon = 1e-6  # To prevent division by zero or log of zero
    tensor = torch.clamp(tensor, epsilon, 1 - epsilon)
    return torch.log(tensor / (1 - tensor))
