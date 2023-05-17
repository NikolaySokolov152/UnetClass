import torch
from torch.nn import BCELoss, MSELoss
from torch.nn import functional as F

@torch.jit.script
def softsign_with_logits(y_hat : torch.Tensor, y_true : torch.Tensor, epsilon : float) -> torch.Tensor:
    z = 1 + torch.abs(y_hat)
    output1 = torch.log(z + y_hat)
    output2 = torch.log(z - y_hat)

    return torch.mean(-torch.log(0.5) - y_true * output1 - (1 - y_true) * output2 + torch.log(z))

@torch.jit.script
def inv_square_with_logits(y_hat : torch.Tensor, y_true : torch.Tensor, epsilon : float) -> torch.Tensor:
    z = 1 + y_hat ** 2
    z_sqrt = torch.sqrt(z)
    output1 = torch.log(z + z_sqrt)
    output2 = torch.log(z - y_hat * z_sqrt)
    return torch.mean(-torch.log(0.5) - y_true * output1 - (1 - y_true) * output2 + torch.log(z))

@torch.jit.script
def bce_with_logits(y_hat : torch.Tensor, y_true : torch.Tensor, epsilon : float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(y_hat, y_true)

class DiceLoss():
    def __init__(self):
        self.smooth = 0.00001
    def __call__(self, y_pred, y_true):
        numerator = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum()
        return 1 - (2 * numerator + self.smooth) / (denominator + self.smooth)

class LossMulticlass():
    def __init__(self, num_classes, loss):
        self.num_classes = num_classes
        self.loss_class = loss

    def __call__(self, y_pred, y_true):
        mitric = 0.0
        for i in range(self.num_classes):
            mitric += self.loss_class(y_pred[:,i,:,:], y_true[:,i,:,:])

        return mitric/self.num_classes

class LossMulticlassEps():
    def __init__(self, num_classes, loss):
        self.num_classes = num_classes
        self.loss_class = loss

    def __call__(self, y_pred, y_true, eps):
        mitric = 0.0
        for i in range(self.num_classes):
            mitric += self.loss_class(y_pred[:,i,:,:], y_true[:,i,:,:], eps)

        return mitric/self.num_classes

def getLossByName(name_loss, num_classes = 1, last_activation = "sigmoid_activation"):
    calculate_stable_loss = False
    if name_loss == "BCELoss":
        if last_activation == "sigmoid_activation":
            #print("Using bce_with_logits (numerical stable)")
            calculate_stable_loss = True
            loss_func = bce_with_logits
        elif last_activation == "softsign_activation":
            #print("Using softsign_with_logits (numerical stable)")
            loss_func = softsign_with_logits
            calculate_stable_loss = True
        elif last_activation == "inv_square_root_activation":
            #print("Using inv_square_with_logits (numerical stable)")
            loss_func = inv_square_with_logits
            calculate_stable_loss = True
        else:
            #print("Using BCELoss (not necessarily numerical stable)")
            loss_func = BCELoss()

    elif name_loss == "MSELoss":
        loss_func = MSELoss()
    elif name_loss == "DiceLoss":
        loss_func = DiceLoss()

    elif name_loss == 'BCELossMulticlass':
        if last_activation == "sigmoid_activation":
            #print("Using bce_with_logits (numerical stable)")
            loss_func = LossMulticlassEps(num_classes, bce_with_logits)
            calculate_stable_loss = True
        elif last_activation == "softsign_activation":
            #print("Using softsign_with_logits (numerical stable)")
            loss_func = LossMulticlassEps(num_classes, softsign_with_logits)
            calculate_stable_loss = True
        elif last_activation == "inv_square_root_activation":
            #print("Using inv_square_with_logits (numerical stable)")
            loss_func = LossMulticlassEps(num_classes, inv_square_with_logits)
            calculate_stable_loss = True
        else:
            #print("Using BCELoss (not necessarily numerical stable)")
            loss_func = LossMulticlass(num_classes, BCELoss())
    elif name_loss == 'DiceLossMulticlass':
        loss_func = LossMulticlass(num_classes, DiceLoss())
    elif name_loss == 'MSELossMulticlass':
        loss_func = LossMulticlass(num_classes, MSELoss())
    else:
        raise Exception(f"LOSS NAME ERROR ! I NO LOSS FUNCTION {name_loss}")

    return loss_func, calculate_stable_loss
