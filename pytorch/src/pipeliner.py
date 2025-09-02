import sys
if not __name__ == "__main__":
    sys.path.append("src/")

import copy
import joblib
import numpy as np
import os
import time
import torch

#from tqdm import tqdm
from tqdm.auto import tqdm

from activation_function import getActivationFunctionByName
from history             import History
from losses              import getLossFunUnion
from metrics             import getCalcMetric, getMetricsAndName
from models              import getModelClassByName
from optimize_parser     import getOptimizerByName
from prepare_data        import getPrepareDataFunction


EPSILON = 1e-7

class Pipeliner:
    def __init__(self,
                 model_type,
                 last_activation_name="sigmoid_activation",
                 num_classes=1,
                 num_channel=1,  # Для диффузионки кол-во каналов для входа и выхода одинаковое
                 device="cpu",
                 silence_mode=False,
                 task_mode="segmentation",  #["segmentation", "img2img", "diffusion"]
                 hidden_params=None,
                 classnames=None,
                 description=None
                 ):

        self.num_classes = num_classes
        self.num_channels = num_channel
        self.model = getModelClassByName(model_type)(n_channels=self.num_channels,
                                                     n_classes=self.num_classes)
        self.last_activation_name = last_activation_name
        self.last_activation_fun = getActivationFunctionByName(last_activation_name)
        self.device = device
        self.silence_mode = silence_mode
        self.task_mode = task_mode
        self.hidden_params = hidden_params
        self.classnames = classnames
        self.log_description = {} if description is None else {"Description": description}

    def set_discrioption(self, dict_data):
        self.log_description = copy.deepcopy(dict_data)

    def add_data_to_discription(self, key, value):
        if not key in self.log_description.keys():
            if isinstance(value, list) or \
               isinstance(value, dict):
                self.log_description[key] = value
            else:
                self.log_description[key] = [value]

        else:
            if isinstance(value, list):
                self.log_description[key] += value
            elif isinstance(value, dict):
                self.log_description[key] |= value
            else:
                self.log_description[key].append(value)

    def incriment_discription_counter(self, key):
        if not key in self.log_description.keys():
            self.log_description[key] = 1
        else:
            self.log_description[key] += 1

    def getLossFun(self, losses, EPSILON, weights_classes):
        '''
        # Перенаправляет вызов выбора вычисляющей фунеции getLossFun без дублирования классовых методов
        '''
        return getLossFunUnion(losses,
                               EPSILON,
                               self.last_activation_name,
                               self.num_classes,
                               self.device,
                               weights_classes=weights_classes)
    def train_block(self,
                    tqdm_train_loop,
                    optimizer,
                    now_lr=None,
                    use_train_metric=True):

        self.model.train()

        train_metric = torch.zeros(len(self.metric_names_list))
        loss_val = 0.0
        start_train_time = time.time()
        for epoch_train_iteration, train_data in enumerate(tqdm_train_loop):
            # print(inputs.shape, targets.shape)
            inputs, targets, batch_statistic = self.prepareDataFun(train_data, self.use_count_pixel_balance)
            # print(inputs.shape, targets.shape)
            # epoch_view_image(epoch_train_iteration, inputs, targets)

            # PREDICT WITHOUT ACTIVATION !!!!
            # так как в случае диффузионок подется и данные и временая метка, то в случае модели с 2 входами
            # input будет содержать 2 значения, а с одним один и распоковываться на месте.
            outputs = self.model(*inputs)
            loss = self.lossesFunction(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now_loss_val = loss.item()
            loss_val += now_loss_val

            # epoch_save_image(epoch, epoch_train_iteration, inputs, targets) ############################################################

            with torch.no_grad():
                if use_train_metric:
                    # ADD LAST ACTIVATION
                    outputs = self.last_activation_fun(outputs, EPSILON)
                    # outputs = outputs.to(device)
                    now_iteration_metric = self.calcMetric(outputs, targets)
                    train_metric = train_metric.add(now_iteration_metric)
                    # print(calculate_metrics(metrics, outputs, targets))
                    metrics_view = {}
                    for i, metric_name in enumerate(self.metric_names_list):
                        metrics_view[metric_name] = now_iteration_metric[i].numpy()

                    str_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in metrics_view.items()])
                else:
                    str_metrics_view = 'No use'

            tqdm_train_loop.set_description(f"\tTrain metric: {str_metrics_view}, loss= {now_loss_val:.4f}")
            tqdm_train_loop.set_postfix_str(f"lr= {now_lr}")

        end_train_time = time.time()
        self.train_history.update_train_work_time(end_train_time - start_train_time)

        # Save epoch train inform
        metrics_view_epoch = {}
        if use_train_metric:
            for i, metric_name in enumerate(self.metric_names_list):
                metrics_view_epoch[metric_name] = np.round(train_metric[i].numpy() / len(tqdm_train_loop),
                                                           4)
                self.train_history.update_train_metric(metric_name, metrics_view_epoch[metric_name])

            str_mean_train_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in metrics_view_epoch.items()])
        else:
            str_mean_train_metrics_view = 'No use'

        last_train_loss = loss_val / len(tqdm_train_loop)

        return last_train_loss, str_mean_train_metrics_view


    def validation_block(self,
                         tqdm_valid_loop):

        self.model.eval()

        val_metric = torch.zeros(len(self.metric_names_list))
        val_loss_val = 0.0
        with torch.no_grad():
            start_valid_time = time.time()
            for epoch_valid_iteration, validate_data in enumerate(tqdm_valid_loop):
                inputs, targets, batch_statistic = self.prepareDataFun(validate_data, self.use_count_pixel_balance)

                outputs = self.model(*inputs)
                # outputs = outputs.to(device)
                loss = self.lossesFunction(outputs, targets)
                # ADD LAST ACTIVATION
                outputs = self.last_activation_fun(outputs, EPSILON)
                val_loss_val += loss.item()

                val_metric = val_metric.add(self.calcMetric(outputs, targets))
                val_metrics_view = {}
                for i, metric_name in enumerate(self.metric_names_list):
                    val_metrics_view[metric_name] = np.round(val_metric[i].numpy() / (epoch_valid_iteration + 1), 4)

                str_val_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in val_metrics_view.items()])
                tqdm_valid_loop.set_description(
                    f"\tValidation metric: {str_val_metrics_view},"
                    f" loss= {(val_loss_val / (epoch_valid_iteration + 1)):.4f}")

            end_valid_time = time.time()
            self.train_history.update_validation_work_time(end_valid_time - start_valid_time)

            # Save epoch validation inform
            metrics_view_epoch_val = {}
            for i, metric_name in enumerate(self.metric_names_list):
                metrics_view_epoch_val[metric_name] = np.round(
                    val_metric[i].numpy() / len(tqdm_valid_loop), 4)
                self.train_history.update_val_metric(metric_name, metrics_view_epoch_val[metric_name])

            last_validation_loss = val_loss_val / len(tqdm_valid_loop)
            self.train_history.update_validation_loss(last_validation_loss)
            str_mean_val_metrics_view = ', '.join([f'{k} : {v:.4f}' for k, v in metrics_view_epoch_val.items()])

        return last_validation_loss, str_mean_val_metrics_view

    def save_model(self):
        if self.save_model_mode == "all":
            torch.save(self.model, self.model_name + '.pt')
        elif self.save_model_mode == "weights_only":
            self.temp_best_model_weights = copy.deepcopy(self.model.state_dict())
            torch.save(self.temp_best_model_weights, self.model_name + '.pth')
        elif self.save_model_mode == "weights_only_after_finish":
            # copy.deepcopy перестрахует что веса не перетрутся, сохранятся в ОЗУ и этот вызов не повлияет на модель.
            self.temp_best_model_weights = copy.deepcopy(self.model.state_dict())
            print("The model weights is saved in RAM")
        else:
            msg = (f'ERROR! Save model mode "{self.save_model_mode}" is not correct !'
                   f' save_model_mode can be only "all", "weights_only" and "weights_only_after_finish" !')
            raise AttributeError(msg)

    def load_model_weights(self, model_weights):
        self.model.load_state_dict(model_weights)

    def load_model_weights_path(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def load_model_by_path(self, model_path):
        self.model = torch.load(model_path)

    def save_pipeliner(self):
        if self.save_pipeliner_mode == "no_save":
            if self.save_model_mode == "weights_only_after_finish":
                torch.save(self.temp_best_model_weights, self.model_name + '.pth')
                print("The model weights is saved in ROM")

        elif self.save_pipeliner_mode == "save_pipeliner":
            if self.save_model_mode == "weights_only_after_finish" or \
               self.save_model_mode == "weights_only":
                # save packege
                joblib.dump((self.model.__class__.__name__,
                                   self.last_activation_name,
                                   self.num_classes,
                                   self.num_channels,
                                   self.task_mode,
                                   self.classnames,
                                   self.temp_best_model_weights,
                                   self.hidden_params,
                                   self.log_description),
                            self.model_name + "_pipeline.pkl")
                if self.save_model_mode == "weights_only":
                    print("Delete the file with weights, because it duplicates the weight in the general pipeline file.")
                    os.remove(self.model_name + '.pth')
            else:
                # save packege
                joblib.dump((self.model.__class__.__name__,
                                   self.last_activation_name,
                                   self.num_classes,
                                   self.num_channels,
                                   self.task_mode,
                                   self.classnames,
                                   None,
                                   self.hidden_params,
                                   self.log_description),
                            self.model_name + "_pipeline.pkl")
        else:
            raise AttributeError(f'save_pipeliner_mode "{self.save_pipeliner_mode}" is not defined!')

    @staticmethod
    def load_pipeliner(pipeliner_path):
        if os.path.isfile(pipeliner_path):
            (model_name,
             last_activation_name,
             num_classes,
             num_channel,
             task_mode,
             classnames,
             weights,
             hidden_params,
             log_discription) = joblib.load(pipeliner_path)

            work_pipeliner = Pipeliner(model_type=model_name,
                                       last_activation_name=last_activation_name,
                                       num_classes=num_classes,
                                       num_channel=num_channel,
                                       task_mode=task_mode,
                                       classnames=classnames,
                                       hidden_params=hidden_params)
            work_pipeliner.set_discrioption(log_discription)

            if weights is None:
                print("WARNING!!! Pipeliner model weights are initialized randomly!")
            else:
                work_pipeliner.load_model_weights(weights)
            return work_pipeliner
        else:
            msg = f"File path error!!! Pipeliner file '{pipeliner_path}' is not founded."
            raise FileNotFoundError(msg)

    def getBalancyFlag(self, class_statistic, train_args):
        ####################################################################################################################### ПЕРВАЯ ВЕРСИЯ БАЛАНСИРОВКИ ##########################################
        use_count_pixel_balance = False
        p_weights = None
        if class_statistic is not None:
            if train_args["balancing_parameters"]['use_p_class_balance']:
                self.train_history.history["class_statistic"] = self.train_history.numpy_dic_to_dic_list(class_statistic)
                p_weights = []
                for statistic_weights in class_statistic["p_classes"]:
                    p_weights.append(1 - statistic_weights)
                p_weights = np.array(p_weights)
                self.train_history.history["class_statistic"]["work_p_classes_weights"] = p_weights.tolist()

            if train_args["balancing_parameters"]["use_count_pixel_balance"]:
                use_count_pixel_balance = True

        return use_count_pixel_balance, p_weights

    def checkEarlyStop(self):
        last_6_value = self.train_history.history["loss"]["validation_loss"][-6:]
        counter_no_changes = 0
        if len(last_6_value) != 6:
            return False
        for i in range(5):
            if last_6_value[i] - last_6_value[-1] < 0.0001:
                counter_no_changes+=1
        if counter_no_changes == 5:
            print("Нет изменений по последним 5 значениям более чем на 0.0001, но")
        print("WARNING!!! checkEarlyStop не реализован и на данный момент не работает!")
        return False

    def train(self,
              data,
              num_epoch,
              optimizer_name,
              metric_names,
              losses,
              lr_scheduler,
              use_validation,
              use_train_metric,
              train_args,
              model_name=None,
              device = None,
              save_model_mode="weights_only_after_finish",  # [ "all", "weights_only", "weights_only_after_finish"]
              save_pipeliner_mode="save_pipeliner"): # ["no_save", "save_pipeliner"]

        self.incriment_discription_counter("train_count")
        self.add_data_to_discription("train_log", "Train started")
        start_train_time = time.time()

        # Блок инициализации
        if device is not None:
            self.device = device

        self.model_name = self.model.__class__.__name__ if model_name is None else model_name
        self.save_model_mode = save_model_mode
        self.save_pipeliner_mode = save_pipeliner_mode

        minimum_validation_error = np.finfo(np.float32).max

        self.prepareDataFun = getPrepareDataFunction(self.task_mode, train_args)
        self.model.to(self.device)
        optimizer = getOptimizerByName(optimizer_name, self.model)

        self.train_history = History()

        if hasattr(data, "gen_train"):
            dataset_info = data.get_dataset_info()
            self.train_history.history["generator info"] = {
                "number of images": dataset_info["number of images"],
                "number of titles": dataset_info["number of titles"]
            }
            generator=data
        else:
            msg = "ERROR. Unknown data generator type."
            raise Exception(msg)

        self.classnames = generator.list_class_name
        self.metric_funs_list, self.metric_names_list = getMetricsAndName(metric_names,
                                                                          self.num_classes,
                                                                          self.classnames)
        self.train_history.init_metrics_history(self.metric_names_list)
        self.calcMetric = getCalcMetric(self.metric_funs_list)

        self.use_count_pixel_balance, p_weights = self.getBalancyFlag(generator.class_statistic, train_args)
        self.lossesFunction = self.getLossFun(losses,
                                              EPSILON,
                                              weights_classes=p_weights)
        self.train_history.add_value("train_args", {
            "num_epoch": num_epoch,
            "optimizer_name": optimizer_name,
            "use_validation": use_validation,
            "use_train_metric": use_train_metric,
            "lr_scheduler": lr_scheduler if isinstance(lr_scheduler, str) else lr_scheduler.__name__,
        })

        # Блок вычислений
        print("Start train model", flush=True)
        for epoch in range(num_epoch):
            print(f"\nEpoch {epoch + 1} / {num_epoch}:", flush=True)

            if lr_scheduler is not None:
                now_lr = lr_scheduler(epoch)
                optimizer.param_groups[0]['lr'] = now_lr
                self.train_history.update_val_lr(now_lr)
            else:
                now_lr = optimizer.param_groups[0]['lr']

            try:
                cmd_size = os.get_terminal_size().columns
            except Exception:
                cmd_size = 200

            # Train loop
            if not self.silence_mode:
                time.sleep(0.2) # чтобы tqdm не печатал вперед print
            tqdm_train_loop = tqdm(generator.gen_train,
            #tqdm_train_loop=tqdm(my_data_generator.getTrainDataLoaderPytorch(0),
                                   desc="\t",
                                   ncols=cmd_size-len(f"lr= {now_lr}")-3,
                                   file=sys.stdout,
                                   colour="GREEN",
                                   disable=self.silence_mode)
            # ncols изменен, чтобы при set_postfix_str не было переноса на новую строку
            # desc изменен,чтобы не было 0% в начале

            last_train_loss, str_mean_train_metrics_view = self.train_block(tqdm_train_loop,
                                                                            optimizer,
                                                                            now_lr,
                                                                            use_train_metric=use_train_metric)

            self.train_history.update_train_loss(last_train_loss)
            print(
                f"\tTrain {epoch + 1} of {num_epoch} results: "
                f"mean_epoch_metric: {str_mean_train_metrics_view}, "
                f" mean_epoch_loss: {round(last_train_loss, 4)}\n",
                flush=True)

            # Validation loop
            if use_validation and len(generator.gen_valid) > 0:

                tqdm_valid_loop = tqdm(generator.gen_valid,
                                       desc="\t",
                                       ncols=cmd_size - len(f"lr= {now_lr}") - 3,
                                       file=sys.stdout,
                                       colour="GREEN",
                                       disable=self.silence_mode)

                if not self.silence_mode:
                    time.sleep(0.2)  # чтобы tqdm не печатал вперед print

                last_validation_loss, str_mean_val_metrics_view = self.validation_block(tqdm_valid_loop)

                print(f"\tValidation {epoch + 1} of {num_epoch} results: " +\
                      f"mean_epoch_metric: {str_mean_val_metrics_view}, " +\
                      f"mean_epoch_loss: {np.round(last_validation_loss, 4)}", flush=True)


                # Save the model with the best loss for validation
                if last_validation_loss <= minimum_validation_error or epoch == 0:
                    self.train_history.update_model_saving_epoch(epoch + 1)
                    self.save_model()
                    print(
                        f"Loss of the model has decreased. {minimum_validation_error} to {last_validation_loss}. " +\
                        f"Model {self.model_name} was saving",
                        flush=True)
                    minimum_validation_error = last_validation_loss
                else:
                    print(
                        f"best result loss: {minimum_validation_error}, " +\
                        f"on epoch: {self.train_history.update_model_saving_epoch.get_last_value()}")

            else:
                if use_validation and not len(generator.gen_valid) > 0:
                    print("WARNING!!! Validation dataset error ! I use train data !")

                # Save the model with the best loss for train
                if last_train_loss <= minimum_validation_error or epoch == 0:
                    self.train_history.update_model_saving_epoch(epoch + 1)
                    self.save_model()
                    print(
                        f"Loss of the model has decreased. {minimum_validation_error} to {last_train_loss}." +\
                        f"Model {self.model_name} was saving",
                        flush=True)
                    minimum_validation_error = last_train_loss
                else:
                    print(
                        f"best result loss: {minimum_validation_error}, " +\
                        f"on epoch: {self.train_history.update_model_saving_epoch.get_last_value()}")

            if self.checkEarlyStop():
                break
            generator.on_epoch_end()

        print("Finish Train")
        self.save_pipeliner()

        end_train_time = time.time()
        self.train_history.add_value("all_train_time", end_train_time - start_train_time)
        self.add_data_to_discription("train_log", "Train finished")

        return self.train_history.history

    def predict(self, generator, device=None, eps=EPSILON, desc_tqdm="\t\tPredict"):
        self.incriment_discription_counter("predict_count")
        if self.task_mode == "diffusion":
            msg=f'ERROR!!! task_mode "{self.task_mode}" is not compatible with "predict" method!'
            raise AttributeError(msg)

        # Блок инициализации
        if device is not None:
            self.device = device

        self.model.to(self.device)
        self.model.eval()
        result = []

        with torch.no_grad():
            tqdm_test_loop = tqdm(generator,
                                  #leave=False,
                                  #file=sys.stdout,
                                  desc=desc_tqdm,
                                  colour="GREEN",
                                  disable=self.silence_mode,
                                  leave=False,
                                  position=1,
                                  ncols=80
                                  )
            for inputs in tqdm_test_loop:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                # ADD LAST ACTIVATION
                outputs = self.last_activation_fun(outputs, eps)

                result.append(outputs.detach().cpu().permute(0, 2, 3, 1).numpy())

        return result

    def generate(self):
        if self.task_mode != "diffusion":
            msg=f'ERROR!!! task_mode "{self.task_mode}" is not compatible with "generate" method!'
            raise AttributeError(msg)

        raise NotImplemented("Pipeliner generate method ещё не готов !")
        return data

