
class Dice():
    def __init__(self):
        self.smooth = 0.00001
    def __call__(self, y_pred, y_true):
        numerator = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum()
        return (2 * numerator + self.smooth) / (denominator + self.smooth)

class DiceMultilabel():
    def __init__(self, num_classes = 2):
        self.num_classes = num_classes
    def __call__(self, y_pred, y_true):
        dice = 0
        for index in range(self.num_classes):
            dice += Dice()(y_true[:, index, :, :], y_pred[:, index, :, :])
        return dice / self.num_classes  # taking average

'''
def universal_dice_coef_multilabel(numLabels):
    def dice_coef(y_true, y_pred):
        dice = 0
        for index in range(numLabels):
            dice += dice_coef_calcucate(y_true[:, :, :, index], y_pred[:, :, :, index])
        return dice / numLabels  # taking average

    return dice_coef

def dice_coef_calcucate(y_true, y_pred):
    smooth = 0.000001
    y_true_f = K.flatten(y_true)  # (K.mean(y_true, axis = 0))
    y_pred_f = K.flatten(y_pred)  # (K.mean(y_pred, axis = 0))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dc_0(y_true, y_pred):
    return dice_coef_calcucate(y_true[:, :, :, 0], y_pred[:, :, :, 0])
def dc_1(y_true, y_pred):
    return dice_coef_calcucate(y_true[:, :, :, 1], y_pred[:, :, :, 1])
def dc_2(y_true, y_pred):
    return dice_coef_calcucate(y_true[:, :, :, 2], y_pred[:, :, :, 2])
def dc_3(y_true, y_pred):
    return dice_coef_calcucate(y_true[:, :, :, 3], y_pred[:, :, :, 3])
def dc_4(y_true, y_pred):
    return dice_coef_calcucate(y_true[:, :, :, 4], y_pred[:, :, :, 4])
def dc_5(y_true, y_pred):
    return dice_coef_calcucate(y_true[:, :, :, 5], y_pred[:, :, :, 5])

def universal_dice_coef_multilabel_arr(numLabels):
    view_classes = [dc_0, dc_1, dc_2, dc_3, dc_4, dc_5]
    return view_classes[:numLabels]
'''
