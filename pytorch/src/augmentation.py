import albumentations as albu
from torchvision.transforms import v2
from torchvision.transforms import Resize
import torch
import numpy as np

# albumentations block
def create_transform_albu(aug_dict, transform_data, augment=True):
    list_compose = []

    if augment:
        if "noise_limit" in aug_dict.keys() and aug_dict["noise_limit"] != 0:
            list_compose.append(albu.GaussNoise(p=0.9, var_limit=aug_dict["noise_limit"] / 256, per_channel=False))

        if "horizontal_flip" in aug_dict.keys() and aug_dict["horizontal_flip"]:
            list_compose.append(albu.HorizontalFlip(p=0.5))
        if "vertical_flip" in aug_dict.keys() and aug_dict["vertical_flip"]:
            list_compose.append(albu.VerticalFlip(p=0.5))

        list_compose.append(
            albu.ShiftScaleRotate(p=0.5, rotate_limit=0, scale_limit=0, shift_limit_x=aug_dict["width_shift_range"],
                                  shift_limit_y=aug_dict["height_shift_range"], border_mode=aug_dict["fill_mode"]))

        if "brightness_shift_range" in aug_dict.keys() and "contrast_shift_range" in aug_dict.keys():
            list_compose.append(albu.RandomBrightnessContrast(p=0.33,
                                                              brightness_limit=aug_dict["brightness_shift_range"],
                                                              contrast_limit=aug_dict["contrast_shift_range"]))
        if "gamma_limit" in aug_dict.keys():
            list_compose.append(albu.RandomGamma(p=0.33, gamma_limit=aug_dict["gamma_limit"]))

        if "rotation_range" in aug_dict.keys():
            list_compose.append(albu.Rotate(p=0.5, limit=aug_dict["rotation_range"], border_mode=aug_dict["fill_mode"]))

        if "zoom_range" in aug_dict.keys():
            list_compose.append(albu.RandomResizedCrop(p=1, height=transform_data.target_size[1],
                                                       width=transform_data.target_size[0],
                                                       scale=(1 - aug_dict["zoom_range"], 1 + aug_dict["zoom_range"]),
                                                       ratio=(1 - aug_dict["zoom_range"], 1 + aug_dict["zoom_range"])))

    list_compose.append(
        albu.Resize(height=transform_data.target_size[1], width=transform_data.target_size[0]))

    # add more https://albumentations.ai/docs/api_reference/full_reference/

    return albu.Compose(list_compose)

def create_albu_transform(aug_dict, transform_data, augment=True):
    transform = create_transform_albu(aug_dict, transform_data, augment)
    def transform_albu(img, masks):
        composed = transform(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        X = torch.from_numpy(np.array(aug_img)).permute(2, 0, 1)
        y = torch.from_numpy(np.array(aug_masks)).permute(2, 0, 1)
        return X, y

    return transform_albu

# pytorch block

class AddGaussianNoise:
    def __init__(self, mean=0., std=1., p=0.5, alpha=0.5, mode="Normal"):
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("`alpha` should be a floating point value in the interval [0.0, 1.0].")
        self.std = std
        self.mean = mean
        self.p = p
        self.alpha = alpha
        self.mode = mode

    def __call(self, tensor):
        if torch.rand(1) < self.p:
            if self.mode == "Normal":
                res_tensor = tensor * (1 - self.alpha) + (
                            torch.randn_like(tensor) * self.std + self.mean) * self.alpha
                return res_tensor
            elif self.mode == "ConstZero":
                noise_tensor = torch.randn_like(tensor) * self.std + self.mean
                res_tensor = tensor.clone()
                res_tensor[tensor != 0] = tensor[tensor != 0] * (1 - self.alpha) + noise_tensor[
                    tensor != 0] * self.alpha
                return res_tensor
            else:
                raise Exception(f"ERROR choise mode {self.mode}")
        else:
            return tensor

    def __call__(self, *args):
        tensor = args[0]
        result = self.__call(tensor)
        if len(args) > 1:
            return result, *args[1:]
        else:
            return result

    def forward(self, *args):
        return self.__call__(args)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, p={2}, alpha={3}, mode={4})'.format(self.mean, self.std, self.p, self.alpha, self.mode)

class VarRandomRotation:
    def __init__(self, degrees, fill, p=1.):
        self.rotate = v2.RandomRotation(degrees=degrees, fill=fill)
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        self.p = p

    def __call__(self, *args):
        if torch.rand(1) < self.p:
            return self.rotate(args)
        else:
            return args

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

    def forward(self, *args):
        return self.__call__(args)

class VarRandomShift:
    def __init__(self,
                 translate,
                 fill,
                 p=1.):
        self.shift = v2.RandomAffine(degrees=0, translate=translate, fill=fill)
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        self.p = p

    def __call__(self, *args):
        if torch.rand(1) < self.p:
            return self.shift(args)
        else:
            return args

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

    def forward(self, *args):
        return self.__call__(args)

class OneArgBrightnessContrast:
    def __init__(self, brightness, contrast, p=0.5):
        self.changer = v2.RandomPhotometricDistort(p=p,
                                                   brightness=(1-brightness, 1+brightness),
                                                   contrast=(1-contrast,1+contrast))

    def __call__(self, *args):
        tensor = args[0]
        result = self.changer(tensor)
        if len(args) > 1:
            return result, *args[1:]
        else:
            return result

    def forward(self, *args):
        return self.__call__(args)

    def __repr__(self):
        return self.__class__.__name__

class Var90Rotate:
    def __init__(self,
                 fill,
                 p=1.):
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        self.p = p
        self.fill = fill

    def __call__(self, *args):
        res = []
        if torch.rand(1) < self.p:
            for tensor in args:
                # 270 чтобы повернуть по часовой на 90
                res.append(v2.functional.rotate(tensor, angle=270, fill=self.fill))
            return res
        else:
            return args

    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, fill={1})'.format(self.p, self.fill)

    def forward(self, *args):
        return self.__call__(args)


def create_torch_transform(aug_dict, transform_data, augment=True):
    list_compose = []
    list_compose.append(v2.ToDtype(torch.float32, scale=True))

    if augment:
        if "horizontal_flip" in aug_dict.keys() and aug_dict["horizontal_flip"]:
            list_compose.append(v2.RandomHorizontalFlip(p=0.5))
        if "vertical_flip" in aug_dict.keys() and aug_dict["vertical_flip"]:
            list_compose.append(v2.RandomVerticalFlip(p=0.5))

        if "p_rotate_90" in aug_dict.keys():
            list_compose.append(Var90Rotate(p=aug_dict["p_rotate_90"], fill=aug_dict["fill_mode"]))


        if "brightness_shift_range" in aug_dict.keys() and "contrast_shift_range" in aug_dict.keys():
            list_compose.append(OneArgBrightnessContrast(p=0.33,
                                                         brightness=aug_dict["brightness_shift_range"],
                                                         contrast=aug_dict["contrast_shift_range"]))

        if "width_shift_range" in aug_dict.keys() or "height_shift_range" in aug_dict.keys():
            if not "width_shift_range" in aug_dict.keys():
                aug_dict["width_shift_range"] = 0
            if not "height_shift_range" in aug_dict.keys():
                aug_dict["height_shift_range"] = 0

            list_compose.append(VarRandomShift(p=0.5,
                                               translate =[aug_dict["width_shift_range"],
                                                           aug_dict["height_shift_range"]],
                                               fill=aug_dict["fill_mode"]))

        if "rotation_range" in aug_dict.keys():
            list_compose.append(VarRandomRotation(p=0.5,
                                                  degrees=aug_dict["rotation_range"],
                                                  fill=aug_dict["fill_mode"]))

        if "zoom_range" in aug_dict.keys():
            list_compose.append(v2.RandomResizedCrop(size=(transform_data.target_size[1], transform_data.target_size[0]),
                                                     scale=(1 - aug_dict["zoom_range"], 1 + aug_dict["zoom_range"]),
                                                     ratio=(1 - aug_dict["zoom_range"], 1 + aug_dict["zoom_range"]),
                                                     antialias=True))

        if "noise_limit" in aug_dict.keys() and aug_dict["noise_limit"] != 0:
            list_compose.append(AddGaussianNoise(p=0.9, alpha=aug_dict["noise_limit"] / 256))

    list_compose.append(v2.Resize(size=(transform_data.target_size[1], transform_data.target_size[0]), antialias=True))

    return v2.Compose(list_compose)



def create_transform(aug_dict, transform_data, augment=True):
    try:
        torch_transform = create_torch_transform(aug_dict, transform_data, augment)
        return torch_transform
    except Exception as ex:
        print(f"Warning!!! Torch augmentation have a problem:")
        print(f"\tException name: {type(ex).__name__}\n\tat line {ex.__traceback__.tb_lineno}\n\tfile: {__file__}\n\texception: {ex}")
        raise Exception("I don't work")
        print("I use more slow albumenation version")
        return create_albu_transform(aug_dict, transform_data, augment)