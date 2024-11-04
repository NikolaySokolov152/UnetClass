
class Dice:
    def __name__(self):
        return "Dice"

    def __init__(self):
        self.smooth = 0.00001
    def __call__(self, y_pred, y_true):
        numerator = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum()
        return (2 * numerator + self.smooth) / (denominator + self.smooth)

class DiceMultilabel:
    #def __name__(self):
    #    return "DiceMultilabel"

    def __init__(self, num_classes = 2):
        self.num_classes = num_classes
        self.d1metric = Dice()
    def __call__(self, y_pred, y_true):
        dice = 0
        for index in range(self.num_classes):
            dice += self.d1metric(y_true[:, index, :, :], y_pred[:, index, :, :])
        return dice / self.num_classes  # taking average

def DiceMultilabelClasses (num_classes = 2):
    d1metric = Dice()

    def dc_0(y_true, y_pred):
        return d1metric(y_true[:, 0, :, :], y_pred[:, 0, :, :])

    def dc_1(y_true, y_pred):
        return d1metric(y_true[:, 1, :, :], y_pred[:, 1, :, :])

    def dc_2(y_true, y_pred):
        return d1metric(y_true[:, 2, :, :], y_pred[:, 2, :, :])

    def dc_3(y_true, y_pred):
        return d1metric(y_true[:, 3, :, :], y_pred[:, 3, :, :])

    def dc_4(y_true, y_pred):
        return d1metric(y_true[:, 4, :, :], y_pred[:, 4, :, :])

    def dc_5(y_true, y_pred):
        return d1metric(y_true[:, 5, :, :], y_pred[:, 5, :, :])

    view_classes = [dc_0,
                    dc_1,
                    dc_2,
                    dc_3,
                    dc_4,
                    dc_5]

    return view_classes[:num_classes]
