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

import diffusers
from diffusers import UNet2DModel
import torch.optim as optim
from diffusers import DDPMScheduler

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

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def merge_inputs_target(inputs, targets):
    if targets is not None:
        return torch.cat((inputs, targets), 1)
    else:
        return inputs

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

######################################################################################################################################## костыль хардкода
def change_model(device, num_classes):
    model = UNet2DModel(
        sample_size=256,  # the target image resolution
        in_channels=num_classes,  # the number of input channels, 3 for RGB images
        out_channels=num_classes,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    return model, optimizer



def fit_diffusion_Model(my_data_generator,
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

    ##################################################################################################################### доделать для 3 каналов rgb
    if train_mode=="diffusion":
        num_classes=1+my_data_generator.num_classes
    else:
        num_classes=my_data_generator.num_classes
    model, optimizer=change_model(device, num_classes)

    CalculateLosses = get_work_loss(losses, EPSILON, last_activation, num_classes)

    print("Start train model", flush=True)

    steps_denoise = train_args["steps_denoise"]
    noise_scheduler = DDPMScheduler(num_train_timesteps=steps_denoise)

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

        #model.train()
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
            t = torch.randint(0, steps_denoise, (my_data_generator.transform_data.batch_size,), device=device)
            ############################################################################################################ костыль
            if my_data_generator.transform_data.mode_mask == "no_mask":
                targets=None
            inputs=merge_inputs_target(inputs,targets)

            noise = torch.randn_like(inputs)

            inputs_noise = noise_scheduler.add_noise(inputs, noise, t).requires_grad_()

            # PREDICT WITHOUT ACTIVATION !!!!
            outputs = model(inputs_noise, t, return_dict=False)[0]
            loss = CalculateLosses(outputs, noise)

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
                    now_iteration_metric = calculate_metrics(metrics, outputs, noise)
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
            #model.eval()
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
                    t = torch.randint(0, steps_denoise, (my_data_generator.transform_data.batch_size,), device=device)
                    ############################################################################################################ костыль
                    if my_data_generator.transform_data.mode_mask == "no_mask":
                        targets = None
                    inputs = merge_inputs_target(inputs, targets)
                    noise = torch.randn_like(inputs)

                    inputs_noise = noise_scheduler.add_noise(inputs, noise, t)

                    # PREDICT WITHOUT ACTIVATION !!!!
                    outputs = model(inputs_noise, t, return_dict=False)[0]
                    outputs = outputs.to(device)
                    loss = CalculateLosses(outputs, noise)
                    # ADD LAST ACTIVATION
                    outputs = globals()[last_activation](outputs, EPSILON)
                    val_loss_val += loss.item()

                    val_metric = val_metric.add(calculate_metrics(metrics, outputs, noise))
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
        #torch.save(model, f"log_model/epoch_{epoch+1}_{modelName}.pt")
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
