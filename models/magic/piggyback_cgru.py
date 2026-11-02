"""Masked GRU backbone sharing MAGIC's recurrent model lifecycle."""

from models.magic.piggyback_rnn import PiggyBackRNN
from models.magic.piggyback_layers import ElementWiseGRU


class PiggyBackGRU(PiggyBackRNN):
    """Three-gate GRU with shared heads, historical masks and expansion handling."""

    recurrent_name = "gru"
    recurrent_layer = ElementWiseGRU
