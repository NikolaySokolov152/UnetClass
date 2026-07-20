import os
import random

def get_name_and_ext(filename):
    name, ext = os.path.splitext(filename)  # получаем расширение
    if ext == ".gz":
        name, ext2 = os.path.splitext(name)
        ext = ext2 + ext
    return name, ext

def add_mods_in_file_name(prefix_file, file_path, file_suffix):
    filename = os.path.basename(file_path)  # получаем имя файла с расширением
    name, ext = get_name_and_ext(filename)
    directory = os.path.dirname(file_path)  # получаем путь к директории
    return os.path.join(directory, prefix_file + name + file_suffix + ext)

def hierarchical_dir_search(path, dir, found_type, ignore_suffixes=tuple()):
    now_dir_path = os.path.join(path, dir)
    name_list = []
    if os.path.isdir(now_dir_path):
        for name in os.listdir(now_dir_path):
            name_list += hierarchical_dir_search(now_dir_path, name, found_type, ignore_suffixes)
    else: # if 'dir' is file
        filename, ext = get_name_and_ext(dir)
        if ext in found_type and not filename.endswith(ignore_suffixes):
            return [now_dir_path]
    return name_list

def search_data_by_type(path_to_img_dir, found_type, ignore_data_suffix):
    names_of_dir = [name for name in os.listdir(path_to_img_dir) if name.endswith((found_type))]
    if len(names_of_dir) == 0:
        print(f'No found file type "{found_type}" in dir "{path_to_img_dir}". Start of hierarchical search')
        img_names_all = []
        for dir in os.listdir(path_to_img_dir):
            img_names_all += hierarchical_dir_search(path_to_img_dir, dir, found_type,
                                                     tuple(ignore_data_suffix))
    else:
        img_names_all = names_of_dir
    return img_names_all

def name_list_sequence_checking(name_list, min_count):
    if len(name_list) < min_count:
        return None
    else:



def get_part_of_list(full_list, proportion_of_dataset, proportion_taking_type="random"):
    if proportion_of_dataset < 1:
        len_names = len(full_list)

        work_num_img = int(round(len_names * proportion_of_dataset))
        # if not self.silence_mode:
        print(f"I take {work_num_img} of {len_names} imagess, proportion of dataset: {proportion_of_dataset}")

        if proportion_taking_type == "random":
            return random.sample(full_list, work_num_img)
        elif proportion_taking_type == "sequentially":
            return full_list[:work_num_img]
        else:
            print("proportion_taking_type not found!")
            raise AttributeError(
                f"proportion_taking_type '{proportion_taking_type}' is unknown! proportion_taking_type may be only 'random' or 'sequentially' !")
    else:
        return full_list

def get_common_path(save_report_path):
    if os.name == 'nt':  # for Windows
        save_report_path = os.path.abspath(save_report_path)
        if save_report_path.startswith(u"\\\\"):
            save_report_path = u"\\\\?\\UNC\\" + save_report_path[2:]
        else:
            save_report_path = u"\\\\?\\" + save_report_path

    return save_report_path