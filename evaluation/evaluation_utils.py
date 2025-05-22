import os
import torch


def compute_model_size(model):
    sfx = str(hex(id(model)))
    torch.save(model.state_dict(), f"temp{sfx}.p")
    size = os.path.getsize(f"temp{sfx}.p") / 1e6
    os.remove(f"temp{sfx}.p")
    return size


def compute_magic_size(models):
    size = 0
    for model in models[:-1]:
        sfx = str(hex(id(model)))
        torch.save(model.state_dict(), f"temp{sfx}.p")
        size = size + os.path.getsize(f"temp{sfx}.p") / 1e6
        os.remove(f"temp{sfx}.p")
    sfx = str(hex(id(models[-1])))
    torch.save(models[-1].state_dict(), f"temp{sfx}.p")
    size2 = os.path.getsize(f"temp{sfx}.p") / 1e6
    os.remove(f"temp{sfx}.p")
    return (size, size2)
