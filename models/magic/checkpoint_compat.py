"""Adapters for previous recurrent APIs and checkpoint attribute names."""


def recurrent_checkpoint_state(state):
    state = state.copy()
    for old_name, new_name in (
        ("GRU_mask_weights", "recurrent_mask_weights"),
        ("Linear_mask_weights", "linear_mask_weights"),
    ):
        if old_name in state:
            value = state.pop(old_name)
            state.setdefault(new_name, value)
    return state


def legacy_argument(value, alias):
    if value is not None and alias is not None:
        raise ValueError("Specify only one value for the argument and its legacy alias")
    return value if alias is None else alias
