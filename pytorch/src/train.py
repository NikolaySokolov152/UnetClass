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

#########################################################################################################запихать всё в класс

def get_work_loss(losses, eps, last_fun_activation_name, num_classes, weights=None):
    num_losses = len(losses)
    if weights is None:
        weights = torch.full(torch.Size([num_losses]), 1/num_losses)

    losses_list = []
    for loss_name in losses:
        losses_list.append(getLossByName(loss_name, num_classes=num_classes, last_activation=last_fun_activation_name))

    def calculate_losses (outputs, targets):
        loss_result = torch.zeros(num_losses)
        for i, (loss_func, calculate_stable_loss) in enumerate(losses_list):
            if calculate_stable_loss:
                loss_result[i] = loss_func(outputs, targets, eps)
            else:
                outputs = globals()[last_fun_activation_name](outputs, eps)
                loss_result[i] = loss_func(outputs, targets)
        return torch.matmul(loss_result, weights)

    return calculate_losses

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

def prepareSegmentationData():
    def prepare_data(inputs, targets):
        return inputs.requires_grad_(), targets
    return prepare_data

def prepareImg2ImgData():
    def prepare_data(inputs, targets):
        return inputs.requires_grad_(), targets
    return prepare_data

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def prepareDiffusionData(train_args):
    def prepare_data(inputs, targets=None):
        steps_denoise = train_args["steps_denoise"]

        # постепенно усиливать шум
        if not "work_alpha" in train_args:
            print("create work_alpha")
            train_args["work_alpha"]=train_args["alpha"]

        alpha=train_args["work_alpha"]


        t = torch.randint(steps_denoise - 1, (inputs.shape[0],), dtype=torch.float32, device=inputs.get_device())
        t_alpha = t*alpha

        work_betta = torch.sqrt(torch.sub(1, t_alpha))
        work_alpha = torch.sqrt(t_alpha)

        # поканальное по batch умножнение на число шага (для каждого отдельного шаг свой)
        denoise_targets = inputs * work_betta.view((-1, *np.ones(inputs.dim() - 1, dtype=int))) +\
                          torch.randn_like(inputs) * work_alpha.view((-1, *np.ones(inputs.dim() - 1, dtype=int)))

        # зашумить на 1 шаг
        input_img = denoise_targets * np.sqrt(1.-alpha) + torch.randn_like(denoise_targets) * np.sqrt(alpha)

        return input_img, denoise_targets

    def diff_prepare_data(inputs, targets=None):
        steps_denoise = train_args["steps_denoise"]
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, steps_denoise)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        t = torch.randint(steps_denoise - 1, (inputs.shape[0],), dtype=torch.float32, device=inputs.get_device())
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, inputs.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, inputs.shape
        )

    return prepare_data

def getPrepareDataFunction(mode, train_args):
    if mode == "segmentation":
        return prepareSegmentationData()
    elif mode == "img2img":
        return prepareImg2ImgData()
    elif mode == "diffusion":
        return prepareDiffusionData(train_args)
    else:
        raise Exception(f"ERROR ! Train mode don't know {mode}")

def epoch_save_image(epoch, epoch_train_iteration, inputs, targets):
    input=inputs.detach().cpu().permute(0, 2, 3, 1).numpy()
    output=targets.detach().cpu().permute(0, 2, 3, 1).numpy()

    batch=inputs.shape[0]

    save_input_path="save_image/input/"
    save_output_path="save_image/output/"

    if not os.path.isdir(save_input_path):
        print("создаю save_input_path:" + save_input_path)
        os.makedirs(save_input_path)
    if not os.path.isdir(save_output_path):
        print("создаю save_output_path:" + save_output_path)
        os.makedirs(save_output_path)

    for b in range(batch):
        cv2.imwrite(os.path.join(save_input_path, f"epoch_{epoch}_iter_{epoch_train_iteration}_input_{b}.png"), input[b] * 255)
        cv2.imwrite(os.path.join(save_output_path, f"epoch_{epoch}_iter_{epoch_train_iteration}_output_{b}.png"), output[b] * 255)



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
             silence_mode = False,
             use_validation = True,
             use_train_metric = True,
             train_mode="segmentation",
             train_args=None):

    history_metrics, val_history_metrics=initHistoryMetric(metrics)

    history_losses = []
    val_history_losses = []
    learning_rate = []
    train_work_time = []
    validation_work_time = []

    minimum_validation_error = np.finfo(np.float32).max
    model_saving_epoch = 0

    PrepereFunction=getPrepareDataFunction(train_mode, train_args)
    CalculateLosses = get_work_loss(losses, EPSILON, last_activation, my_data_generator.num_classes)

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
        if not silence_mode:
            time.sleep(0.2) # чтобы tqdm не печатал вперед print
        tqdm_train_loop = tqdm(my_data_generator.gen_train,
        #tqdm_train_loop=tqdm(my_data_generator.getTrainDataLoaderPytorch(0),
                               desc="\t",
                               ncols=cmd_size-len(f"lr= {now_lr}")-3,
                               file=sys.stdout,
                               colour="GREEN",
                               disable=silence_mode)
        # ncols изменен, чтобы при set_postfix_str не было переноса на новую строку
        # desc изменен,чтобы не было 0% в начале

        start_train_time = time.time()
        for epoch_train_iteration, (inputs, targets) in enumerate(tqdm_train_loop):
            inputs, targets = PrepereFunction(inputs, targets)

            # PREDICT WITHOUT ACTIVATION !!!!
            outputs = model(inputs)
            loss = CalculateLosses(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now_loss_val = loss.item()
            loss_val += now_loss_val

            #epoch_save_image(epoch, epoch_train_iteration, inputs, targets) ############################################################

            with torch.no_grad():
                if use_train_metric:
                    # ADD LAST ACTIVATION
                    outputs = globals()[last_activation](outputs, EPSILON)
                    outputs = outputs.to(device)
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

        end_train_time = time.time()
        train_work_time.append(end_train_time-start_train_time)

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
                if not silence_mode:
                    time.sleep(0.2)  # чтобы tqdm не печатал вперед print
                tqdm_valid_loop = tqdm(my_data_generator.gen_valid,
                                       desc="\t",
                                       ncols=cmd_size-len(f"lr= {now_lr}")-3,
                                       file=sys.stdout,
                                       colour="GREEN",
                                       disable=silence_mode)

                start_valid_time = time.time()
                for epoch_valid_iteration, (inputs, targets) in enumerate(tqdm_valid_loop):
                    inputs, targets = PrepereFunction(inputs, targets) #inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    outputs = outputs.to(device)
                    loss = CalculateLosses(outputs, targets)
                    # ADD LAST ACTIVATION
                    outputs = globals()[last_activation](outputs, EPSILON)
                    val_loss_val += loss.item()

                    val_metric = val_metric.add(calculate_metrics(metrics, outputs, targets))
                    val_metrics_view = {}
                    for i, metric in enumerate(metrics):
                        val_metrics_view[metric.__class__.__name__] = np.round(val_metric[i].numpy() /(epoch_valid_iteration + 1), 4)

                    str_val_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in val_metrics_view.items()])
                    tqdm_valid_loop.set_description(f"\tValidation metric: {str_val_metrics_view}, loss= {(val_loss_val / (epoch_valid_iteration + 1)):.4f}")

                end_valid_time = time.time()

                # Save epoch validation inform
                metrics_view_epoch_val = {}
                for i, metric in enumerate(metrics):
                    metrics_view_epoch_val[metric.__class__.__name__] = np.round(val_metric[i].numpy() / len(my_data_generator.gen_valid), 4)
                    val_history_metrics[metric.__class__.__name__].append(metrics_view_epoch_val[metric.__class__.__name__])
                val_history_losses.append(np.round(val_loss_val / len(my_data_generator.gen_valid), 4))

                str_mean_val_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in metrics_view_epoch_val.items()])
                print(f"\tValidation {epoch + 1} of {num_epoch} results: mean_epoch_metric: {str_mean_val_metrics_view}, mean_epoch_loss: {val_history_losses[-1]}", flush=True)

                # Save the model with the best loss for validation
                if val_history_losses[-1] <= minimum_validation_error or epoch == 0:
                    model_saving_epoch = epoch + 1
                    torch.save(model, modelName + '.pt')
                    print(f"Loss of the model has decreased. {minimum_validation_error} to {val_history_losses[-1]}. model {modelName} was saving", flush=True)
                    minimum_validation_error = val_history_losses[-1]
                else:
                    print(f"best result loss: {minimum_validation_error}, on epoch: {model_saving_epoch}")

                validation_work_time.append(end_valid_time-start_valid_time)
        else:
            if use_validation and not len(my_data_generator.gen_valid) > 0:
                print("WARNING!!! Validation dataset error ! I use train data !")

            # Save the model with the best loss for train
            if history_losses[-1] <= minimum_validation_error or epoch == 0:
                model_saving_epoch = epoch + 1
                torch.save(model, modelName + '.pt')
                print(f"Loss of the model has decreased. {minimum_validation_error} to {history_losses[-1]}. model {modelName} was saving", flush=True)
                minimum_validation_error = history_losses[-1]
            else:
                print(f"best result loss: {minimum_validation_error}, on epoch: {model_saving_epoch}")


        ##################################################################################################################################### нужно доработать
        torch.save(model, f"log_model/epoch_{epoch+1}_{modelName}.pt")
        if "work_alpha" in train_args.keys():
            print(train_args["work_alpha"])
            alpha_linear_curve_coef = (0.002 - train_args["alpha"])/num_epoch
            train_args["work_alpha"]=train_args["alpha"] + epoch*alpha_linear_curve_coef
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
