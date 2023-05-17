import sys
if not __name__ == "__main__":
    sys.path.append("src/")

from losses import *
from models import *
import torch

import os
import numpy as np
from tqdm import tqdm
import time

EPSILON = 1e-7

def calculate_losses (losses, outputs, targets, eps, last_fun_activation_name, weights = None):
    num_losses = len(losses)
    num_classes = targets.shape[1]

    if weights is None:
        weights = torch.full(torch.Size([num_losses]), 1/num_losses)
    loss_result = torch.zeros(num_losses)
    for i, loss_name in enumerate(losses):
        loss_func, calculate_stable_loss = getLossByName(loss_name, num_classes=num_classes, last_activation=last_fun_activation_name)
        if calculate_stable_loss:
            loss_result[i] = loss_func(outputs, targets, eps)
        else:
            outputs = globals()[last_fun_activation_name](outputs, EPSILON)
            loss_result[i] = loss_func(outputs, targets)

    return torch.matmul(loss_result, weights)

def calculate_metrics(metrics, predict, targets):
    num_metrics = len(metrics)
    metrics_result = torch.zeros(num_metrics)
    for i, metric in enumerate(metrics):
        metrics_result[i] = metric(predict, targets)

    return metrics_result

def initHistoryMetric(metrics):
    history_metrics = {}
    val_history_metrics = {}

    for metric in metrics:
        history_metrics[metric.__class__.__name__] = []
        val_history_metrics[metric.__class__.__name__] = []

    return history_metrics, val_history_metrics

def fitModel(my_data_generator,
             model,
             last_activation,
             num_epoch,
             device,
             optimizer,
             metrics,
             losses,
             modelName,
             lr_scheduler = None,
             use_validation = True,
             use_train_metric = True):

    history_metrics, val_history_metrics=initHistoryMetric(metrics)

    history_losses = []
    val_history_losses = []
    learning_rate = []
    train_work_time = []
    validation_work_time = []

    minimum_validation_error = np.finfo(np.float32).max
    model_saving_epoch = 0

    print("Start train model", flush=True)

    for epoch in range(num_epoch):
        print(f"\nEpoch {epoch+1} / {num_epoch}:", flush=True)

        if lr_scheduler is not None:
            now_lr = lr_scheduler(epoch)
            optimizer.param_groups[0]['lr'] = now_lr
            learning_rate.append(now_lr)

        try:
            cmd_size = os.get_terminal_size().columns
        except Exception:
            cmd_size = 200

        model.train()
        train_metric = torch.zeros(len(metrics))
        loss_val = 0.0
        # Train loop
        time.sleep(0.2) # чтобы tqdm не печатал вперед print
        tqdm_train_loop = tqdm(my_data_generator.gen_train,
                               desc="\t",
                               ncols=cmd_size-len(f"lr= {now_lr}")-3,
                               file=sys.stdout,
                               colour="GREEN")
        # ncols изменен, чтобы при set_postfix_str не было переноса на новую строку
        # desc изменен,чтобы не было 0% в начале
        for epoch_train_iteration, (inputs, targets) in enumerate(tqdm_train_loop):
            inputs, targets = inputs.requires_grad_().to(device), targets.to(device)

            # PREDICT WITHOUT ACTIVATION !!!!
            outputs = model(inputs)
            loss = calculate_losses(losses, outputs, targets, EPSILON, last_activation)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now_loss_val = loss.item()
            loss_val += now_loss_val

            with torch.no_grad():
                if use_train_metric:
                    # ADD LAST ACTIVATION
                    outputs = globals()[last_activation](outputs, EPSILON)
                    outputs = outputs.requires_grad_().to(device)
                    now_iteration_metric = calculate_metrics(metrics, outputs, targets)
                    train_metric = train_metric.add(now_iteration_metric)
                    #print(calculate_metrics(metrics, outputs, targets))
                    metrics_view = {}
                    for i, metric in enumerate(metrics):
                        metrics_view[metric.__class__.__name__] = now_iteration_metric[i].numpy()

                    str_metrics_view = ', '.join([f'{k} : {v:.4f}' for k,v in metrics_view.items()])
                else:
                    str_metrics_view = 'No use'

            tqdm_train_loop.set_description(f"\tTrain metric: {str_metrics_view}, loss= {now_loss_val:.4f}")
            if lr_scheduler is not None:
                tqdm_train_loop.set_postfix_str(f"lr= {now_lr}")

        train_work_time.append(tqdm_train_loop.format_dict['elapsed'])

        # Save epoch train inform
        metrics_view_epoch = {}
        if use_train_metric:
            for i, metric in enumerate(metrics):
                metrics_view_epoch[metric.__class__.__name__] = np.round(train_metric[i].numpy() / len(my_data_generator.gen_train), 4)
                history_metrics[metric.__class__.__name__].append(metrics_view_epoch[metric.__class__.__name__])

            str_mean_train_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in metrics_view_epoch.items()])
        else:
            str_mean_train_metrics_view = 'No use'

        history_losses.append(round(loss_val /len(my_data_generator.gen_train), 4))
        print(f"\tTrain {epoch+1} of {num_epoch} results: mean_epoch_metric: {str_mean_train_metrics_view}, mean_epoch_loss: {history_losses[-1]}\n", flush=True)

        # Validation loop
        if use_validation and len(my_data_generator.gen_valid) > 0:
            model.eval()
            with torch.no_grad():
                val_metric = torch.zeros(len(metrics))
                val_loss_val = 0.0
                time.sleep(0.2)  # чтобы tqdm не печатал вперед print
                tqdm_valid_loop = tqdm(my_data_generator.gen_valid,
                                       desc="\t",
                                       ncols=cmd_size-len(f"lr= {now_lr}")-3,
                                       file=sys.stdout,
                                       colour="GREEN")
                for epoch_valid_iteration, (inputs, targets) in enumerate(tqdm_valid_loop):
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    outputs = outputs.to(device)
                    loss = calculate_losses(losses, outputs, targets, EPSILON, last_activation)
                    # ADD LAST ACTIVATION
                    outputs = globals()[last_activation](outputs, EPSILON)
                    val_loss_val += loss.item()

                    val_metric = val_metric.add(calculate_metrics(metrics, outputs, targets))
                    val_metrics_view = {}
                    for i, metric in enumerate(metrics):
                        val_metrics_view[metric.__class__.__name__] = np.round(val_metric[i].numpy() /(epoch_valid_iteration + 1), 4)

                    str_val_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in val_metrics_view.items()])
                    tqdm_valid_loop.set_description(f"\tValidation metric: {str_val_metrics_view}, loss= {(val_loss_val / (epoch_valid_iteration + 1)):.4f}")

                # Save epoch validation inform
                metrics_view_epoch_val = {}
                for i, metric in enumerate(metrics):
                    metrics_view_epoch_val[metric.__class__.__name__] = np.round(val_metric[i].numpy() / len(my_data_generator.gen_valid), 4)
                    val_history_metrics[metric.__class__.__name__].append(metrics_view_epoch_val[metric.__class__.__name__])
                val_history_losses.append(np.round(val_loss_val / len(my_data_generator.gen_valid), 4))

                str_mean_val_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in metrics_view_epoch_val.items()])
                print(f"\tValidation {epoch + 1} of {num_epoch} results: mean_epoch_metric: {str_mean_val_metrics_view}, mean_epoch_loss: {val_history_losses[-1]}", flush=True)

                # Save the model with the best loss for validation
                if val_history_losses[-1] < minimum_validation_error or epoch == 0:
                    model_saving_epoch = epoch + 1
                    torch.save(model, modelName + '.pt')
                    print(f"Loss of the model has decreased. {minimum_validation_error} to {val_history_losses[-1]}. model {modelName} was saving", flush=True)
                    minimum_validation_error = val_history_losses[-1]
                else:
                    print(f"best result loss: {minimum_validation_error}, on epoch: {model_saving_epoch}")

                validation_work_time.append(tqdm_valid_loop.format_dict['elapsed'])
        else:
            if use_validation and not len(my_data_generator.gen_valid) > 0:
                print("WARNING!!! Validation dataset error ! I use train data !")

            # Save the model with the best loss for train
            if history_losses[-1] < minimum_validation_error or epoch == 0:
                model_saving_epoch = epoch + 1
                torch.save(model, modelName + '.pt')
                print(f"Loss of the model has decreased. {minimum_validation_error} to {history_losses[-1]}. model {modelName} was saving", flush=True)
                minimum_validation_error = history_losses[-1]
            else:
                print(f"best result loss: {minimum_validation_error}, on epoch: {model_saving_epoch}")

        my_data_generator.on_epoch_end()

    print("Finish Train")

    history = {"metrics": history_metrics,
               "loss": history_losses,
               "train_work_time": train_work_time,
               "model_saving_epoch": model_saving_epoch
               }
    if use_validation:
        history["val_metrics"] = val_history_metrics
        history["val_loss"] = val_history_losses
        history["validation_work_time"] = validation_work_time
    if lr_scheduler is not None:
        history["lr"] = learning_rate

    return history
