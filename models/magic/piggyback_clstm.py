"""Masked LSTM backbone sharing MAGIC's recurrent model lifecycle."""

from models.magic.piggyback_rnn import PiggyBackRNN
from models.magic.piggyback_layers import ElementWiseLSTM


class PiggyBackLSTM(PiggyBackRNN):
    """Four-gate LSTM with shared heads, historical masks and expansion handling."""

    recurrent_name = "lstm"
    recurrent_layer = ElementWiseLSTM
