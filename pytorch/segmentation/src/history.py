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

    def __init__(self, metric_names=None):
        if metric_names is not None:
            self.init_metrics_history(metric_names)
        self.update_train_work_time      = self.Update_after_init("train_work_time", "work_time")
        self.update_validation_work_time = self.Update_after_init("validation_work_time", "work_time")

        self.update_train_loss           = self.Update_after_init("train_loss", "loss")
        self.update_validation_loss      = self.Update_after_init("validation_loss", "loss")

        self.update_val_lr               = self.Update_after_init("lr")
        self.update_model_saving_epoch   = self.Update_after_init("model_saving_epoch")

    def init_metrics_history(self, metrics_name):
        self.history["metrics"] = {}
        self.history["metrics"]["train_metrics"] = {}
        self.history["metrics"]["val_metrics"] = {}

        for i, metric_name in enumerate(metrics_name):
            self.history["metrics"]["train_metrics"][metric_name] = []
            self.history["metrics"]["val_metrics"][metric_name] = []

    def update_train_metrics(self, metric_names, value_metrics):
        for i, metric in enumerate(metric_names):
            self.history["metrics"]["train_metrics"][metric].append(value_metrics[i])
    def update_val_metrics(self, metric_names, value_metrics):
        for i, metric in enumerate(metric_names):
            self.history["metrics"]["val_metrics"][metric].append(value_metrics[i])

    def update_train_metric(self, metric_name, value_metric):
        self.history["metrics"]["train_metrics"][metric_name].append(value_metric)
    def update_val_metric(self, metric_name, value_metric):
        self.history["metrics"]["val_metrics"][metric_name].append(value_metric)

    def numpy_dic_to_dic_list(self, dic):
        return_dic={}
        for key in dic.keys():
            return_dic[key] = dic[key].tolist()
        return return_dic

    def add_value(self, name_dict, value):
        self.history[name_dict] = value