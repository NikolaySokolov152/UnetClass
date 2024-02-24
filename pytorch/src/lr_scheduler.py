# функция изменения lr
#def standart_lr_scheduler(epoch):
#    if epoch < 100:
#        return 0.0001
#    elif epoch < 125:
#        return 0.00005
#    elif epoch < 150:
#        return 0.00001
#    elif epoch < 175:
#        return 0.000005
#    return 0.0000001

def standart_lr_scheduler(epoch):
    if epoch < 20:
        return 0.0001
    elif epoch < 40:
        return 0.00005
    elif epoch < 60:
        return 0.00001
    elif epoch < 80:
        return 0.000005
    return 0.000001

def loss_lr_scheduler(epoch):
    if epoch < 300:
        return 0.0001
    elif epoch < 350:
        return 0.00005
    elif epoch < 400:
        return 0.00001
    elif epoch < 450:
        return 0.000005
    return 0.000001

def loss_mix_lr_scheduler(epoch):
    if epoch < 200:
        return 0.0001
    elif epoch < 225:
        return 0.00005
    elif epoch < 250:
        return 0.00001
    elif epoch < 275:
        return 0.000005
    return 0.000001

def lr_scheduler_hard(epoch):
    if epoch < 20:
        return 0.1
    elif epoch < 40:
        return 0.01
    elif epoch < 60:
        return 0.001
    elif epoch < 80:
        return 0.0001
    return 0.00001