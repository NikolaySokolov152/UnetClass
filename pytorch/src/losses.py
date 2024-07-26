import torch
from torch.nn import BCELoss, MSELoss
from torch.nn import functional as F

import numpy as np
import cv2

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

@torch.jit.script
class DiceLoss:
    def __init__(self):
        self.smooth = 0.00001
    def __call__(self, y_pred : torch.Tensor, y_true : torch.Tensor):
        numerator = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum()
        return 1 - (2 * numerator + self.smooth) / (denominator + self.smooth)

@torch.jit.script
class LossMulticlassDice:
    def __init__(self, num_classes:int):
        self.num_classes:int = num_classes
        self.loss_class = DiceLoss()

    def __call__(self, y_pred : torch.Tensor, y_true: torch.Tensor):
        mitric = 0.0
        for i in range(self.num_classes):
            mitric += self.loss_class(y_pred[:,i,:,:], y_true[:,i,:,:])

        return mitric/self.num_classes

#@torch.jit.script
class LossMulticlass:
    def __init__(self, num_classes:int, loss):
        self.num_classes:int = num_classes
        self.loss_class = loss

    def __call__(self, y_pred : torch.Tensor, y_true: torch.Tensor):
        mitric = 0.0
        for i in range(self.num_classes):
            mitric += self.loss_class(y_pred[:,i,:,:], y_true[:,i,:,:])

        return mitric/self.num_classes

#@torch.jit.script
class LossMulticlassEps:
    def __init__(self, num_classes:int, loss):
        self.num_classes:int = num_classes
        self.loss_class = loss

    def __call__(self, y_pred, y_true, eps):
        mitric = 0.0
        for i in range(self.num_classes):
            mitric += self.loss_class(y_pred[:,i,:,:], y_true[:,i,:,:], eps)

        return mitric/self.num_classes

@torch.jit.script
class LossDistance2Nearest:
    def __init__(self, num_classes: int):
        self.num_classes: int = num_classes
        self.max_dist: int = 32
        self.overlap_dist: int = 1
        self.smooth:float = 0.00001
        self.error_penalty:int = 1

    def __call__(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        metric = 0.0
        for c in range(self.num_classes):
            conv_true = self.batch_slice_distanse(y_true[:,c,:,:])
            conv_pred = self.batch_slice_distanse(y_pred[:,c,:,:])

            # подсчет 2 дистанций для избавления от ошибок 1 и 2 рода
            dictance_counts = torch.mul(y_pred[:,c,:,:], conv_true).sum()\
                              +torch.mul(y_true[:,c,:,:], conv_pred).sum()

            # smooth - защита деления на 0 (нет класса и нечего не предсказалось)
            batch_mean_mark = y_pred[:, c, :, :].sum() + y_true[:, c, :, :].sum() + self.smooth
            metric += dictance_counts / (2 * batch_mean_mark)

        return metric/self.num_classes

    #optimaze version
    def batch_slice_distanse(self, slice:torch.Tensor):
        map_inv_distanse = torch.clamp(slice, 0, self.error_penalty)
        slice2conv = torch.unsqueeze(slice, 1) # size:(b, 1, h, w)

        kernel_size = 2 * (self.overlap_dist) + 1
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float)
        kernel = kernel.to(slice.get_device())
        kernel_tensor = torch.unsqueeze(torch.unsqueeze(kernel, 0), 0)  # size: (1, 1, k, k)

        for kernel_r_size in range(1, self.max_dist//self.error_penalty):
            # morpthological pytorch
            slice2conv = torch.clamp(torch.nn.functional.conv2d(slice2conv,
                                                                kernel_tensor,
                                                                padding=(self.overlap_dist, self.overlap_dist)),
                                     0,
                                     self.error_penalty)

            map_inv_distanse+=torch.squeeze(slice2conv)

        map_distanse = torch.full(slice.shape, self.max_dist, dtype=torch.float)
        map_distanse = map_distanse.to(slice.get_device())

        map_distanse -= map_inv_distanse
        return map_distanse

@torch.jit.script
def huber(y_pred : torch.Tensor, y_true : torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(y_pred, y_true)

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
        loss_func = LossMulticlassDice(num_classes)
    elif name_loss == 'MSELossMulticlass':
        loss_func = LossMulticlass(num_classes, MSELoss())
    elif name_loss == "LossDistance2Nearest":
        loss_func = LossDistance2Nearest(num_classes)
    elif name_loss == "HuberLoss":
        loss_func = huber
    else:
        raise Exception(f"LOSS NAME ERROR ! I NO LOSS FUNCTION {name_loss}")

    return loss_func, calculate_stable_loss

def get_work_loss(losses, eps, last_fun_activation_name, num_classes, weights=None):
    num_losses = len(losses)
    if weights is None:
        weights = torch.full(torch.Size([num_losses]), 1/num_losses)

    last_activation_fun = globals()[last_fun_activation_name]

    losses_list = []
    for loss_name in losses:
        losses_list.append(getLossByName(loss_name, num_classes=num_classes, last_activation=last_fun_activation_name))

    def calculate_losses (outputs, targets):
        loss_result = torch.zeros(num_losses)
        for i, (loss_func, calculate_stable_loss) in enumerate(losses_list):
            if calculate_stable_loss:
                loss_result[i] = loss_func(outputs, targets, eps)
            else:
                outputs =last_activation_fun(outputs, eps)
                loss_result[i] = loss_func(outputs, targets)
        return torch.matmul(loss_result, weights)

    return calculate_losses
