import os
import torch


def compute_model_size(model):
    sfx = str(hex(id(model)))
    torch.save(model.state_dict(), f"temp{sfx}.p")
    size = os.path.getsize(f"temp{sfx}.p") / 1e6
    os.remove(f"temp{sfx}.p")
    return size

def _get_filtered_state(m):
    manager = m.manager
    curr_task_str = str(manager.curr_task_idx)
    full_state = manager.model.state_dict()
    filtered_state = {}

    for key, tensor in full_state.items():
        if key.startswith('heads.'):
            if key.startswith(f'heads.{curr_task_str}.'):
                filtered_state[key] = tensor
        else:
            filtered_state[key] = tensor
    return filtered_state

def compute_magic_size(models):
    size = 0
    for model in models[:-1]:
        sfx = str(hex(id(model)))
        torch.save(_get_filtered_state(model), f"temp{sfx}.p")
        size = size + os.path.getsize(f"temp{sfx}.p") / 1e6
        os.remove(f"temp{sfx}.p")

    sfx = str(hex(id(models[-1])))
    torch.save(_get_filtered_state(models[-1]), f"temp{sfx}.p")
    size2 = os.path.getsize(f"temp{sfx}.p") / 1e6
    os.remove(f"temp{sfx}.p")

    return (size, size2)
