import numpy as np
import torch
from monai import transforms as mntr
from src.DataStructures3D import CommonTransformData


# pytorch block

def create_3d_torch_transform(aug_dict, target_size, augment):
    list_compose = []
    #list_compose.append(mntr.ToDtype(torch.float32, scale=True))

    if augment:
        if "flip" in aug_dict.keys() and aug_dict["flip"] is True:
            list_compose.append(mntr.RandFlipd(prob=0.5, keys=["image", "label"]))

        if "p_rotate_90" in aug_dict.keys():
            list_compose.append(mntr.RandRotate90d(prob=aug_dict["p_rotate_90"], keys=["image", "label"]))

        if "brightness_shift_range" in aug_dict.keys() and "contrast_shift_range" in aug_dict.keys():

            list_compose.append(mntr.RandAdjustContrast(prob=0.33,
                                    gamma=(-aug_dict["contrast_shift_range"], aug_dict["contrast_shift_range"])))

            list_compose.append(mntr.RandShiftIntensity(prob=0.33,
                                    offsets=aug_dict["brightness_shift_range"]))

        if "width_shift_range" in aug_dict.keys() or\
           "height_shift_range" in aug_dict.keys() or\
           "depth_shift_range" in aug_dict.keys():
            if not "width_shift_range" in aug_dict.keys():
                aug_dict["width_shift_range"] = 0
            if not "height_shift_range" in aug_dict.keys():
                aug_dict["height_shift_range"] = 0
            if not "depth_shift_range" in aug_dict.keys():
                aug_dict["depth_shift_range"] = 0

            list_compose.append(mntr.RandAffined(prob=0.5, keys=["image", "label"],
                                                 translate_range = (aug_dict["depth_shift_range"],
                                                                    aug_dict["height_shift_range"],
                                                                    aug_dict["width_shift_range"]),
                                                 padding_mode = aug_dict["fill_mode"]
                                                 ))

        if "rotation_range" in aug_dict.keys():
            list_compose.append(mntr.RandRotated(prob=0.5, keys=["image", "label"],
                                                range_x=aug_dict["rotation_range"],
                                                range_y=aug_dict["rotation_range"],
                                                range_z=aug_dict["rotation_range"],
                                                padding_mode=aug_dict["fill_mode"]))

        if "zoom_range" in aug_dict.keys():
            #list_compose.append(
            #    mntr.RandomResizedCrop(size=(transform_data.target_size[1], transform_data.target_size[0]),
            #                         scale=(1 - aug_dict["zoom_range"], 1 + aug_dict["zoom_range"]),
            #                         ratio=(1 - aug_dict["zoom_range"], 1 + aug_dict["zoom_range"]),
            #                         antialias=True))

            list_compose.append(mntr.RandZoomd(prob=0.5, keys=["image", "label"],
                                              min_zoom= 1 - aug_dict["zoom_range"],
                                              max_zoom= 1 + aug_dict["zoom_range"]))

        if "noise_limit" in aug_dict.keys() and aug_dict["noise_limit"] != 0:
            list_compose.append(mntr.RandGaussianNoised(keys=["image"], prob=0.8, std= np.sqrt(aug_dict["noise_limit"]/256)/3)) #+-3sigma

    list_compose.append(mntr.Resized(spatial_size=[*target_size], anti_aliasing=True,  keys=["image", "label"]))

    return mntr.Compose(list_compose)


def create_transform(aug_dict, target_size, augment=True):
    '''
    try:
        torch_transform = create_torch_transform(aug_dict, transform_data, augment)
        return torch_transform
    except Exception as ex:
        print(f"Warning!!! Torch augmentation have a problem:")
        print(f"\tException name: {type(ex).__name__}\n\tat line {ex.__traceback__.tb_lineno}\n\tfile: {__file__}\n\texception: {ex}")
        raise Exception("I don't work")
        print("I use more slow albumenation version")
        return create_albu_transform(aug_dict, transform_data, augment)

    '''
    return create_3d_torch_transform(aug_dict, target_size, augment)