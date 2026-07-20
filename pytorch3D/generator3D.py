import torch
import numpy as np
from torch.utils.data import Dataset
from monai.data import MetaTensor
from src.augmentation3D import create_transform


def get_rand_piece_of_data(dataframe:np.ndarray|torch.Tensor, maskframe:np.ndarray|torch.Tensor, shape:list|tuple|np.ndarray|torch.Tensor):
    d, h, w = dataframe.shape
    e_d, e_h, e_w = shape

    start_z = np.random.randint(d-e_d)
    start_y = np.random.randint(h-e_h)
    start_x = np.random.randint(w-e_w)

    piece_data = dataframe[start_z:start_z+e_d,
                           start_y:start_y+e_h,
                           start_x:start_x+e_w]

    piece_mask = mitoframe[start_z:start_z+e_d,
                           start_y:start_y+e_h,
                           start_x:start_x+e_w]
    return piece_data, piece_mask

def get_rand_piece_of_tensor(dataframe:torch.Tensor, maskframe:torch.Tensor, shape:list|tuple|np.ndarray|torch.Tensor, device):
    _, d, h, w = dataframe.shape
    e_d, e_h, e_w = shape

    start_z = torch.randint(d-e_d, (), device=device)
    start_y = torch.randint(h-e_h, (), device=device)
    start_x = torch.randint(w-e_w, (), device=device)

    piece_data = dataframe[:,
                           start_z:start_z+e_d,
                           start_y:start_y+e_h,
                           start_x:start_x+e_w]

    piece_mask = maskframe[:,
                           start_z:start_z+e_d,
                           start_y:start_y+e_h,
                           start_x:start_x+e_w]
    return piece_data, piece_mask

def added_zero_channel_axis(data): # DHW -> DHW1
    return np.expand_dims(data, axis = -1)

def get_device_tensor_from_list_of_numpy(data:np.ndarray, device:str|torch.device):
     return torch.from_numpy(data).to(device).permute(3, 0, 1, 2)

def get_device_tensor_from_channel_stack_of_numpy(data:np.ndarray, device:str|torch.device):
     return torch.from_numpy(data).to(device)

def get_cpu_numpy_from_list_of_tensor(data:torch.Tensor):
    return data.detach().cpu().permute(1, 2, 3, 0).numpy()
    
    
class data_generator(Dataset):
    def __init__(self,
                 datasets:list[list],
                 target_shape,
                 batch_size,
                 count_of_exsamples,
                 aug_dict,
                 device,
                 is_augment=True,
                 mode = 'train',
                 probability_choice_gen = None):

        self.datasets = datasets
        self.batch_size = batch_size
        self.count_of_exsamples = count_of_exsamples
        self.target_shape = target_shape
        self.mode = mode
        self.probability_choice_gen = torch.tensor(probability_choice_gen, device=device) if probability_choice_gen else None
        self.get_dataset_fun = self.get_random_dataset_fun()

        self.binary_mask = True # бинаризация масок при трансформациях
        self.composition = create_transform(aug_dict, self.target_shape, is_augment)
        self.device = device

    def get_random_dataset(self):
        index = torch.multinomial(self.probability_choice_gen, num_samples=1).item()
        return self.datasets[index]
    def get_allow_dataset(self):
        return self.datasets

    def get_random_dataset_fun(self):
        if isinstance(self.datasets[0], list):
            return self.get_random_dataset
        else:
            return self.get_allow_dataset

    def __len__(self):
        return self.count_of_exsamples

    def __getitem__(self, index):
        # if self.inform_generator == "validation generator":
        #print(index)
        if index >= self.__len__():
            raise StopIteration

        b_X = []
        b_y = []
        for b in range(self.batch_size):
            # если есть несколько наборов, то они могут перемешаться в батче
            dataset = self.get_dataset_fun()
            e_x, e_y = get_rand_piece_of_tensor(dataset[0],
                                                dataset[1],
                                                self.target_shape,
                                                self.device)
            b_X.append(e_x)
            b_y.append(e_y)
        
        if self.mode == 'train':
            b_X, b_y = self.batch_transform(b_X, b_y)
            b_X = torch.stack(b_X)
            b_y = torch.stack(b_y)
            return b_X, b_y
        else:
            raise AttributeError('The mode parameter should be set to "train".')

    def random_transform_one_frame(self, data, masks):
        aug_data, aug_masks = self.composition({"image": data, "label": masks}).values()
        if self.binary_mask:
            aug_masks = aug_masks.round()
        return aug_data, aug_masks

    def batch_transform(self, data_batch, masks_batch):
        for i in range(len(data_batch)):
            data_batch[i], masks_batch[i] = self.random_transform_one_frame(data_batch[i], masks_batch[i])
        return data_batch, masks_batch

class PiplinerPaperGen:

    def __init__(self, gen_train, list_class_name = ["mito"]):
        self.gen_train = gen_train
        self.class_statistic=None
        self.list_class_name = list_class_name
        self.gen_valid=None
        
    def __len__(self):
        return len(self.gen_train)

    def on_epoch_end(self):
        pass

    def get_dataset_info(self):
        len_of_datasets = len(self.gen_train.datasets) if isinstance(self.gen_train.datasets[0], list) else 1
        dict_ret = {
                    "number of images": len_of_datasets,
                    "number of titles": self.gen_train.count_of_exsamples
                   }
        return dict_ret