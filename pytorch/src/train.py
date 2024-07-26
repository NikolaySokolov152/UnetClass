import sys

import cv2

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

def calculate_metrics(metrics, predict, targets):
    num_metrics = len(metrics)
    metrics_result = torch.zeros(num_metrics)
    for i, metric in enumerate(metrics):
        metrics_result[i] = metric(predict, targets)

    return metrics_result

def prepareSegmentationData():
    def prepare_data(inputs, targets):
        return inputs.requires_grad_(), targets
    return prepare_data

def random_median_image_fun(img):
    kernel = 2*np.random.randint(1, 6) +1

    img = (img*255).astype(np.uint8)

    if kernel>1:
        ret_image_in = cv2.medianBlur(img, kernel-2)
    else:
        ret_image_in = img
    ret_image_out = cv2.medianBlur(img, kernel)

    return (np.expand_dims(ret_image_in , axis=-1).astype(np.float32)/255,
            np.expand_dims(ret_image_out, axis=-1).astype(np.float32)/255)
def prepareImg2ImgData():
    def prepare_data(inputs, targets):
        cpu_inputs = targets.detach().cpu().permute(0, 2, 3, 1).numpy()
        median_input = []
        median_output = []
        for bs in range(cpu_inputs.shape[0]):
            in_img, out_img = random_median_image_fun(cpu_inputs[bs])
            median_input.append(torch.from_numpy(in_img))
            median_output.append(torch.from_numpy(out_img))
        torch_median_input = torch.stack(median_input).to("cuda").permute(0, 3, 1, 2)
        torch_median_output = torch.stack(median_output).to("cuda").permute(0, 3, 1, 2)
        return torch_median_input.requires_grad_(), torch_median_output
    return prepare_data

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

######################################### НЕДОДЕЛАНО
def prepareDiffusionData(train_args):
    # преподготовка
    steps_denoise = train_args["steps_denoise"]
    beta_start=train_args["beta_start"]
    beta_end=train_args["beta_end"]
    betas = torch.linspace(beta_start, beta_end, steps_denoise)
    alphas = 1. - betas

    #################################################### понять##################################################################
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    '''
    print("betas", betas)
    print("alphas", alphas)

    print("alphas_cumprod", alphas_cumprod)
    print("alphas_cumprod_prev", alphas_cumprod_prev)
    print("sqrt_recip_alphas", sqrt_recip_alphas)

    print("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
    print("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
    print("posterior_variance", posterior_variance)
    '''

    '''
    def prepare_data(inputs, targets=None):
        bs=inputs.shape[0]
        t = torch.randint(steps_denoise - 1, (bs,), dtype=torch.float32, device=inputs.get_device())
        t_alphas = extract(alphas, t, bs)
        t_betas = extract(betas, t, bs)


        work_betta = torch.sqrt(torch.sub(1, t_alpha))
        work_alpha = torch.sqrt(t_alpha)


        # поканальное по batch умножнение на число шага (для каждого отдельного шаг свой)
        denoise_targets = inputs * work_betta.view((-1, *np.ones(inputs.dim() - 1, dtype=int))) +\
                          torch.randn_like(inputs) * work_alpha.view((-1, *np.ones(inputs.dim() - 1, dtype=int)))

        # зашумить на 1 шаг
        noise = torch.randn_like(denoise_targets)
        input_img = denoise_targets * np.sqrt(1.-alpha) + noise*np.sqrt(alpha)

        return input_img, noise

    '''
    ###################################################################################################################### смержить маски и вход в одну пачку
    def diff_prepare_data(inputs, targets=None):
        t = torch.randint(0, steps_denoise, (inputs.shape[0],), device=inputs.get_device())

        if targets is not None:
            inputs_targets = torch.cat((inputs, targets), 1)
        else:
            inputs_targets = inputs

        noise = torch.randn_like(inputs_targets)

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, inputs_targets.shape).to(inputs_targets.get_device())
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, inputs_targets.shape).to(
            inputs_targets.get_device())

        input_img = sqrt_alphas_cumprod_t * inputs_targets + sqrt_one_minus_alphas_cumprod_t * noise

        return input_img.requires_grad_(), noise

    return diff_prepare_data

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

def epoch_view_image(epoch_train_iteration, inputs, targets):
    input=inputs.detach().cpu().permute(0, 2, 3, 1).numpy()
    output=targets.detach().cpu().permute(0, 2, 3, 1).numpy()

    batch=inputs.shape[0]
    for b in range(batch):
        cv2.imshow((f"iter_{epoch_train_iteration}_input_{b}.png"), input[b])
        cv2.imshow((f"iter_{epoch_train_iteration}_output_{b}.png"), output[b])
    cv2.waitKey()

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

    history = History(metrics)

    ##################################################################################################################### запихать параметры генератора в историю тренировки
    history.history["num_images"] = len(my_data_generator.list_img_name)
    history.history["num_gen_repetitions"] = my_data_generator.num_gen_repetitions

    minimum_validation_error = np.finfo(np.float32).max

    PrepereFunction=getPrepareDataFunction(train_mode, train_args)

    num_classes=1+my_data_generator.num_classes if train_mode=="diffusion" else my_data_generator.num_classes

    last_activation_fun = globals()[last_activation]

    CalculateLosses = get_work_loss(losses, EPSILON, last_activation, num_classes)

    print("Start train model", flush=True)

    for epoch in range(num_epoch):
        print(f"\nEpoch {epoch+1} / {num_epoch}:", flush=True)

        if lr_scheduler is not None:
            now_lr = lr_scheduler(epoch)
            optimizer.param_groups[0]['lr'] = now_lr
            history.update_val_lr(now_lr)
        else:
            now_lr = optimizer.param_groups[0]['lr']

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
            #print(inputs.shape, targets.shape)
            ############################################################################################################ костыль
            if my_data_generator.transform_data.mode_mask == "no_mask":
                targets=None
            inputs, targets = PrepereFunction(inputs, targets)
            #print(inputs.shape, targets.shape)
            #epoch_view_image(epoch_train_iteration, inputs, targets)

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
                    outputs = last_activation_fun(outputs, EPSILON)
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
        history.update_train_work_time(end_train_time-start_train_time)

        # Save epoch train inform
        metrics_view_epoch = {}
        if use_train_metric:
            for i, metric in enumerate(metrics):
                metrics_view_epoch[metric.__class__.__name__] = np.round(train_metric[i].numpy() / len(my_data_generator.gen_train), 4)
                history.update_train_metric(metric.__class__.__name__, metrics_view_epoch[metric.__class__.__name__])

            str_mean_train_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in metrics_view_epoch.items()])
        else:
            str_mean_train_metrics_view = 'No use'

        last_train_loss = loss_val /len(my_data_generator.gen_train)
        history.update_train_loss(last_train_loss)
        print(f"\tTrain {epoch+1} of {num_epoch} results: mean_epoch_metric: {str_mean_train_metrics_view}, mean_epoch_loss: {round(last_train_loss, 4)}\n", flush=True)

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
                    outputs = last_activation_fun(outputs, EPSILON)
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
                    history.update_val_metric(metric.__class__.__name__, metrics_view_epoch_val[metric.__class__.__name__])

                last_validation_loss = val_loss_val / len(my_data_generator.gen_valid)
                history.update_validation_loss(last_validation_loss)

                str_mean_val_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in metrics_view_epoch_val.items()])
                print(f"\tValidation {epoch + 1} of {num_epoch} results: mean_epoch_metric: {str_mean_val_metrics_view}, mean_epoch_loss: {np.round(last_validation_loss, 4)}", flush=True)

                # Save the model with the best loss for validation
                if last_validation_loss <= minimum_validation_error or epoch == 0:
                    history.update_model_saving_epoch(epoch + 1)
                    torch.save(model, modelName + '.pt')
                    print(f"Loss of the model has decreased. {minimum_validation_error} to {last_validation_loss}. model {modelName} was saving", flush=True)
                    minimum_validation_error = last_validation_loss
                else:
                    print(f"best result loss: {minimum_validation_error}, on epoch: {history.update_model_saving_epoch.get_last_value()}")

                history.update_validation_work_time(end_valid_time-start_valid_time)
        else:
            if use_validation and not len(my_data_generator.gen_valid) > 0:
                print("WARNING!!! Validation dataset error ! I use train data !")

            # Save the model with the best loss for train
            if last_train_loss <= minimum_validation_error or epoch == 0:
                history.update_model_saving_epoch(epoch + 1)
                torch.save(model, modelName + '.pt')
                print(f"Loss of the model has decreased. {minimum_validation_error} to {last_train_loss}. model {modelName} was saving", flush=True)
                minimum_validation_error = last_train_loss
            else:
                print(f"best result loss: {minimum_validation_error}, on epoch: {history.update_model_saving_epoch.get_last_value()}")


        ##################################################################################################################################### нужно доработать
        #torch.save(model, f"log_model/epoch_{epoch+1}_{modelName}.pt")
        my_data_generator.on_epoch_end()

    print("Finish Train")
    return history.history

class History:
    history = {}

    class Update_after_init:
        def __init__(self, name_param, group_name=None):
            self.work_fun = self._history_init_fun
            self.name_param = name_param
            self.group_name = group_name
            self.save_object = History.history
        def get_last_value(self):
            return self.save_object[self.name_param][-1]
        def _history_init_fun(self, value):
            if self.group_name is not None:
                if not self.group_name in self.save_object.keys():
                    self.save_object[self.group_name] = {}
                self.save_object = self.save_object[self.group_name]

            self.save_object[self.name_param] = [value]
            self.work_fun = self._history_update_fun
        def _history_update_fun(self, value):
            self.save_object[self.name_param].append(value)
        def __call__(self, value):
            self.work_fun(value)

    def __init__(self, metrics=None):
        if metrics is not None:
            self.init_metrics_history(metrics)
        self.update_train_work_time      = self.Update_after_init("train_work_time", "work_time")
        self.update_validation_work_time = self.Update_after_init("validation_work_time", "work_time")

        self.update_train_loss           = self.Update_after_init("train_loss", "loss")
        self.update_validation_loss      = self.Update_after_init("validation_loss", "loss")

        self.update_val_lr               = self.Update_after_init("lr")
        self.update_model_saving_epoch   = self.Update_after_init("model_saving_epoch")

    def init_metrics_history(self, metrics):
        self.history["metrics"] = {}
        self.history["metrics"]["train_metrics"] = {}
        self.history["metrics"]["val_metrics"] = {}

        for i, metric in enumerate(metrics):
            self.history["metrics"]["train_metrics"][metric.__class__.__name__] = []
            self.history["metrics"]["val_metrics"][metric.__class__.__name__] = []

    def update_train_metrics(self, metrics, value_metrics):
        for i, metric in enumerate(metrics):
            self.history["metrics"]["train_metrics"][metric.__class__.__name__].append(value_metrics[i])
    def update_val_metrics(self, metrics, value_metrics):
        for i, metric in enumerate(metrics):
            self.history["metrics"]["val_metrics"][metric.__class__.__name__].append(value_metrics[i])

    def update_train_metric(self, metric_name, value_metric):
        self.history["metrics"]["train_metrics"][metric_name].append(value_metric)
    def update_val_metric(self, metric_name, value_metric):
        self.history["metrics"]["val_metrics"][metric_name].append(value_metric)

'''
#Перенос функции в класс
class ModelTrainer:
    def __init__(self,
                 generator,
                 model,
                 last_activation,
                 num_epoch,
                 device,
                 optimizer,
                 metrics,
                 losses,
                 modelName,
                 lr_scheduler=None,
                 silence_mode=False,
                 use_validation=True,
                 use_train_metric=True,
                 train_mode="segmentation",
                 train_args=None
                 ):
        self.num_epoch=num_epoch
        self.train_args=train_args
        self.modelName=modelName
        self.lr_scheduler=lr_scheduler
        self.use_validation=use_validation
        self.silence_mode=silence_mode
        self.use_train_metric=use_train_metric
        self.train_mode=train_mode
        self.model=model
        self.last_activation=last_activation
        self.device = device
        self.optimizer= optimizer
        self.losses=losses
        self.generator=generator
        self.model_saving_epoch = 0
        self.CalculateLosses = get_work_loss(losses, EPSILON, last_activation, self.num_classes)

        self.num_classes = self.get_num_classes()
        self.history = History(metrics)
        self.minimum_validation_error = np.finfo(np.float32).max
        self.PrepereFunction=self.getPrepareDataFunction()


    def getPrepareDataFunction(self):
        if self.train_mode == "segmentation":
            return prepareSegmentationData()
        elif self.train_mode == "img2img":
            return prepareImg2ImgData()
        elif self.train_mode == "diffusion":
            return prepareDiffusionData(self.train_args)
        else:
            raise Exception(f"ERROR ! Train mode don't know {self.train_mode}")
    def get_num_classes(self):
        return 1+self.generator.num_classes if self.train_mode=="diffusion" else self.generator.num_classes
    def optimizer_update(self, epoch):
        now_lr = self.lr_scheduler(epoch)
        self.optimizer.param_groups[0]['lr'] = now_lr
        self.history.update_val_lr_fun(now_lr)

    def get_max_extend_cmd(self):
        try:
            return os.get_terminal_size().columns
        except Exception:
            return 200

    def __call__(self):
        pass
        ##################################################################################################################### доделать для 3 каналов rgb
        print("Start train model", flush=True)

        for epoch in range(self.num_epoch):
            print(f"\nEpoch {epoch+1} / {self.num_epoch}:", flush=True)

            if self.lr_scheduler is not None:
                self.optimizer_update(epoch)

            cmd_size = self.get_max_extend_cmd()

            model.train()
            train_metric = torch.zeros(len(self.metrics))
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
                ############################################################################################################ костыль
                if my_data_generator.transform_data.mode_mask == "no_mask":
                    targets=None
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

'''