from keras import backend as K

import tensorflow as tf

def universal_dice_coef_multilabel(numLabels):
    def dice_coef(y_true, y_pred):
        dice = 0
        for index in range(numLabels):
            dice += dice_coef_calcucate(y_true[:, :, :, index], y_pred[:, :, :, index])
        return dice / numLabels  # taking average

    return dice_coef

def universal_dice_coef_loss(numLabels):
    def dice_coef_loss(y_true, y_pred):
        dice = 0
        for index in range(numLabels):
            dice += dice_coef_calcucate(y_true[:, :, :, index], y_pred[:, :, :, index])
        return 1 - dice / numLabels  # taking average
    
    return dice_coef_loss

def dice_coef_calcucate(y_true, y_pred):
    smooth = 0.000001
    y_true_f = K.flatten(y_true) #(K.mean(y_true, axis = 0))
    y_pred_f = K.flatten(y_pred) #(K.mean(y_pred, axis = 0))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss_calcucate(y_true, y_pred):
    return 1 - dice_coef_calcucate(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

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

def balanced_cross_entropy(beta):
  def BlnsCE_loss(y_true, y_pred):
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)
    
    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
    return tf.reduce_mean(o)

  return BlnsCE_loss
 