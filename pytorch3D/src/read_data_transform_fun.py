import numpy as np

AVALIBLE_TRANSFORM_FUN = ["dev 255",
                          "mean std",
                          "None",
                          "dev 4096",
                          "min max"]

def to_0_1_format_img(in_img):
    max_val = in_img.max()
    if max_val <= 1:
        return in_img
    else:
        out_img = in_img.astype(np.float32) / 255
        return out_img

def to_0_1_format_img_12bit(in_img):
    max_val = in_img[:,:].max()
    if max_val <= 1:
        return in_img
    else:
        out_img = in_img.astype(np.float32) / 4095
        return out_img

def to_mean_std_format_img(in_img):
    mean = np.mean(in_img)
    std_dev = np.std(in_img)

    if std_dev != 0:
        return (in_img - mean) / std_dev
    else:
        return in_img

def min_max_normalaze(in_img):
    min_val = np.min(in_img)
    max_val = np.max(in_img)

    if min_val == max_val:
        return in_img/max(1, max_val)
    else:
        return (in_img.astype(np.float32)-min_val)/(max_val-min_val)

def to_0_255_format_img(in_img):
    max_val = in_img.max()
    if max_val <= 1:
        out_img = np.round(in_img * 255)
        return out_img.astype(np.uint8)
    else:
        return in_img

def batch_to_numpy_from_torch(tensor):
    return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()

def enumerate_slice_transform(in_img):
    # WARNING ZERO VALUE IS IGNORE (is considered background)
    unique_values = np.unique(in_img)[1:]
    if len(unique_values) == 0:
        return np.zeros((in_img.shape[0], in_img.shape[1], 1), dtype=np.float32)

    if max(unique_values) - min(unique_values) > len(unique_values) * 4:
        print(f"WARNING!!! Too many zero classes! Check the format of the input data masks! Max value: {max(unique_values)}, min value: {min(unique_values)}, number of unique {len(unique_values)}")

    data = np.zeros((in_img.shape[0], in_img.shape[1], int(max(unique_values))), dtype=np.float32)
    for i, unique_value in enumerate(unique_values):
        data[:,:, i][in_img[:,:]==unique_value] = 1
    return data

def get_normalization_fun(name=None):
    if name == "dev 255":
        return to_0_1_format_img
    elif name == "mean std":
        return to_mean_std_format_img
    elif name is None or name == "None": # pass fun
        return lambda x: x
    elif name == "dev 4096":
        return to_0_1_format_img_12bit
    elif name == "min max":
        return min_max_normalaze
    else:
        raise Exception(f"normalization function '{name}' is not define")
