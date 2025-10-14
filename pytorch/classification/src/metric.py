import numpy as np
import h5py
import torch
from tqdm import tqdm



def mask_label_to_class_val(mask):
    if mask.sum() == 0:
        return 0
    else:
        return 1

def numpy_data_to_torch(img, mask, device):
    x = torch.from_numpy(img).to(device).permute(2, 0, 1).to(torch.float32)
    y = torch.tensor([mask_label_to_class_val(mask)]).to(device).to(torch.float32)

    return x, y

def gen_get_of_data(list_of_name):
    for path_to_data in list_of_name:
        with h5py.File(path_to_data, "r") as file:
            img = np.array(file["image"])
            mask = np.array(file["mask"])

            yield img, mask

def dataset_gen(list_of_name, batch_size, device="cpu"):
    num_of_batches = len(list_of_name)//batch_size
    end_batch_size = None if len(list_of_name)%batch_size == 0 else len(list_of_name) - batch_size*num_of_batches

    gen_data = gen_get_of_data(list_of_name)
    
    for i in range(num_of_batches):
        batch_data_img = []
        batch_data_mark = []
        for b in range(batch_size):
            x,y = numpy_data_to_torch(*next(gen_data), device=device)
            batch_data_img.append(x)
            batch_data_mark.append(y)
        yield torch.stack(batch_data_img), torch.stack(batch_data_mark)

    if end_batch_size:
        batch_data_img = []
        batch_data_mark = []
        for b in range(end_batch_size):
            x,y = numpy_data_to_torch(*next(gen_data), device=device)
            batch_data_img.append(x)
            batch_data_mark.append(y)
        yield torch.stack(batch_data_img), torch.stack(batch_data_mark)


def get_loader_by_namelist(namelist, batch_size, device="cpu"):
    return tqdm(dataset_gen(namelist, batch_size, device=device), total=len(namelist)//batch_size)

# Функция вычисления точности top-1
def get_accuracy(data_loader, model, device):
    model.eval()
    tp = 0
    n = 0
    with torch.no_grad():
        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            preds = torch.round(outputs)
            n += labels.size(0)
            tp += (preds == labels).sum()

    return tp / n


def get_accuracy_by_name_list(namelist, model, batch_size=1, device="cpu"):
    return get_accuracy(get_loader_by_namelist(namelist, batch_size, device), model, device)
    
# Функция вычисления точности top-1
def get_accuracy_with_output(data_loader, model, device):
    model.eval()
    tp = 0
    n = 0
    pred_list = []
    with torch.no_grad():
        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            preds = torch.round(outputs)
            pred_list.append(preds.item())
            n += labels.size(0)
            tp += (preds == labels).sum()

    #print(f"result {pred_list}")
    return tp / n, pred_list

def get_accuracy_by_name_list_with_output(namelist, model, batch_size=1, device="cpu"):
    return get_accuracy_with_output(get_loader_by_namelist(namelist, batch_size, device), model, device)