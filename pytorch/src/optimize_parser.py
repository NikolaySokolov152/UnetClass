import torch_optimizer as optim_mod
import torch.optim as optim

# GET OPTIMIZER
AVAILABLE_OPTIMIZER_OPTIONS = ['Adam', 'AdamW', 'RMSprop', 'NovoGrad']

def getOptimizerByName(optimizer_name : str, model):
    if optimizer_name is None or optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=5e-3)
    elif optimizer_name == "NovoGrad":
        optimizer = optim_mod.NovoGrad(model.parameters(), lr=1e-3, weight_decay=1e-3)
    else:
        str_using_optimizer = "' ,'".join(AVAILABLE_OPTIMIZER_OPTIONS)
        msg = f"OPTIMIZER CHOICE ERROR!!! Don't know '{optimizer_name}' optimizer." +\
              f"Now you can only choose: '{str_using_optimizer}' optimizer"
        print(msg)
        raise AttributeError(msg)
    return optimizer