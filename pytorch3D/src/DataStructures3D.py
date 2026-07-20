import numpy as np

def check_init_data_using(class_obj, init_dict):
    class_attr_keys = dir(class_obj)

    list_of_no_using_values = []

    for init_key in init_dict.keys():
        if init_key in class_attr_keys:
            continue
        else:
            list_of_no_using_values.append((init_key, init_dict[init_key]))

    if len(list_of_no_using_values) != 0:
        print(f"WARNING CLASS INIT {class_obj.__class__.__name__}!!! DON'T INIT ATTRIBYTE {', '.join([f'{k} with val {v}' for k,v in list_of_no_using_values])}")


class TailingData:
    def __init__(self,
                 is_tiling=False,
                 tailing_fun="v1"
                 ):
        self.tailing_fun=tailing_fun
        self.is_tiling=False if tailing_fun is None else is_tiling

    def get_is_tiling_flag(self):
        return self.is_tiling

    def __bool__(self):
        return self.is_tiling

    def __str__(self):
        return f"\nTailingData:"+ \
               f"\n\t tailing_fun: {self.tailing_fun}" if self.tailing_fun is not None else "None"+ \
               f"\n\t is_tiling: {self.is_tiling}"

class CommonTransformData:
    target_size: list|tuple|np.ndarray

    def __init__(self, **kwargs
                 ):
        self.target_size = kwargs.get('target_size', (80, 80, 80))  # (W, H) !
        self.batch_size = kwargs.get('batch_size', 1)

        self.type_load_data = kwargs.get('type_load_data', "gray")  # ["3d_data_as_imgs"]
        self.normalization_data_fun = kwargs.get('normalization_data_fun', None)
        self.type_load_target = kwargs.get('type_load_target',
                                           "separated")  # ["separated", "image", "no_mask", "numpy", "nii"]
        self.normalization_mask_fun = kwargs.get('normalization_mask_fun', None)
        self.binary_mask = kwargs.get('binary_mask', False)

        check_init_data_using(self, kwargs)

    def __str__(self):
        return f"\nCommonTransformData:" + \
            f"\n\t main transform:" + \
            f"\n\t\t target_size: {self.target_size}" +\
            f"\n\t\t batch_size: {self.batch_size}" + \
            f"\n\t common dataset data:" + \
            ("" if self.type_load_data is None else f"\n\t\t type_load_data: {self.type_load_data}") + \
            ("" if self.normalization_data_fun is None else f"\n\t\t normalization_data_fun: {self.normalization_data_fun}") + \
            ("" if self.type_load_target is None else f"\n\t\t type_load_target: {self.type_load_target}") + \
            ("" if self.normalization_mask_fun is None else f"\n\t\t normalization_mask_fun: {self.normalization_mask_fun}") +\
            ("" if self.binary_mask is None else f"\n\t\t binary_mask: {self.binary_mask}")


class InfoDirData:
    def __init__(self, **kwargs):
        self.dir_img_name = kwargs.get('dir_img_name', kwargs.get('dir_img_path', "data/train/origin"))
        self.dir_mask_name = kwargs.get('dir_mask_path_without_name', "data/train/")
        self.ignore_data_suffix = kwargs.get('ignore_data_suffix', [])
        self.add_mask_prefix = kwargs.get('add_mask_prefix', '')
        self.add_mask_suffix = kwargs.get('add_mask_suffix', '')
        self.common_dir_path = kwargs.get('common_dir_path', None)

        self.type_load_data = kwargs.get('type_load_data', None)  # ["3d_data_as_imgs"]
        self.normalization_data_fun = kwargs.get('normalization_data_fun', None)
        self.type_load_target = kwargs.get('type_load_target',
                                           None)  # ["separated", "image", "no_mask", "numpy", "nii"]
        self.normalization_mask_fun = kwargs.get('normalization_mask_fun', None)
        self.binary_mask = kwargs.get('binary_mask', None)

        self.cropping_data = kwargs.get('cropping_data', False)
        self.num_gen_repetitions = kwargs.get('num_gen_repetitions', 0)

        check_init_data_using(self, kwargs)

    def __str__(self):
        return f"\nInfoDirData:" + \
            f"\n\t str path data config:" + \
            (f"\n\t\t common_dir_path: {self.common_dir_path}" if self.common_dir_path is not None else "") + \
            f"\n\t\t dir_img_name: {self.dir_img_name}" + \
            (f"\n\t\t dir_mask_name: {self.dir_mask_name}" if self.common_dir_path is None else "") + \
            f"\n\t\t ignore_data_suffix: {self.ignore_data_suffix}" + \
            f"\n\t\t add_mask_prefix: {self.add_mask_prefix}" + \
            f"\n\t\t add_mask_suffix: {self.add_mask_suffix}" + \
            f"\n\t number of data config:" + \
            f"\n\t\t proportion_of_dataset: {self.proportion_of_dataset}" + \
            f"\n\t\t proportion_taking_type: {self.proportion_taking_type}" + \
            f"\n\t\t num_gen_repetitions: {self.num_gen_repetitions}" + \
            f"\n\t mods of data config:" + \
            f"\n\t\t cropping_data: {self.cropping_data}" + \
            ("" if self.type_load_data is None else f"\n\t\t type_load_data: {self.type_load_data}") + \
            ("" if self.normalization_data_fun is None else f"\n\t\t normalization_data_fun: {self.normalization_data_fun}") + \
            ("" if self.type_load_target is None else f"\n\t\t type_load_target: {self.type_load_target}") + \
            ("" if self.normalization_mask_fun is None else f"\n\t\t normalization_mask_fun: {self.normalization_mask_fun}") +\
            ("" if self.binary_mask is None else f"\n\t\t binary_mask: {self.binary_mask}")

    def __repr__(self):
        return self.__str__()

class SaveGeneratorData:
    def __init__(self,
                 save_to_dir=None,
                 save_prefix_image="image_",
                 save_prefix_mask="mask_",
                 **kwargs):
        self.save_to_dir = save_to_dir
        self.save_prefix_image = save_prefix_image
        self.save_prefix_mask = save_prefix_mask

    def __str__(self):
        return str(self.__dict__)

def check_dataset_and_comon_transform_info(info_dir_data:InfoDirData, transform_data: CommonTransformData, atribut, check_none=False):
    val_info = getattr(info_dir_data, atribut)
    val_atribut = getattr(transform_data, atribut) if val_info is None \
        else val_info
    if check_none:
        assert val_atribut is not None
    return val_atribut

