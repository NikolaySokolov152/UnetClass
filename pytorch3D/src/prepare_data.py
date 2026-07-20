import os
import skimage.io as io
import torch

AVAILABLE_PREPARE_OPTIONS = ["segmentation", "img2img", "diffusion"]

from read_data_transform_fun import to_0_255_format_img

#################### TRAIN MODEL BLOCK ############################
def segmentationPrepareFun(data_package, use_count_pixel_balance):  # data_package: [input, target, statistic]
    input_data = (data_package[0].requires_grad_(),)
    target_data = data_package[1]
    balance_data = data_package[2] if use_count_pixel_balance else None

    return input_data, target_data, balance_data

def img2imgPrepareFun(data_package, use_count_pixel_balance):
    '''
    def random_median_image_fun(img):
        kernel = 2 * np.random.randint(1, 6) + 1

        img = (img * 255).astype(np.uint8)

        if kernel > 1:
            ret_image_in = cv2.medianBlur(img, kernel - 2)
        else:
            ret_image_in = img
        ret_image_out = cv2.medianBlur(img, kernel)

        return (np.expand_dims(ret_image_in, axis=-1).astype(np.float32) / 255,
                np.expand_dims(ret_image_out, axis=-1).astype(np.float32) / 255)

    cpu_inputs = data_package[0].detach().cpu().permute(0, 2, 3, 1).numpy()
    median_input = []
    median_output = []
    for bs in range(cpu_inputs.shape[0]):
        in_img, out_img = random_median_image_fun(cpu_inputs[bs])
        median_input.append(torch.from_numpy(in_img))
        median_output.append(torch.from_numpy(out_img))
    torch_median_input = torch.stack(median_input).to("cuda").permute(0, 3, 1, 2)
    torch_median_output = torch.stack(median_output).to("cuda").permute(0, 3, 1, 2)
    return [torch_median_input.requires_grad_()], torch_median_output, data_package[2] if self.use_count_pixel_balance else None
    '''

    return segmentationPrepareFun(data_package, use_count_pixel_balance)

def diffusionPrepareFunWarper(train_arg):

    """
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    ######################################### НЕДОДЕЛАНО
    def prepareDiffusionData(train_args):
        # преподготовка
        steps_denoise = train_args["steps_denoise"]
        beta_start = train_args["beta_start"]
        beta_end = train_args["beta_end"]
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

    def prepare_data(inputs, targets=None):
        bs = inputs.shape[0]
        t = torch.randint(steps_denoise - 1, (bs,), dtype=torch.float32, device=inputs.get_device())
        t_alphas = extract(alphas, t, bs)
        t_betas = extract(betas, t, bs)

        work_betta = torch.sqrt(torch.sub(1, t_alpha))
        work_alpha = torch.sqrt(t_alpha)

        # поканальное по batch умножнение на число шага (для каждого отдельного шаг свой)
        denoise_targets = inputs * work_betta.view((-1, *np.ones(inputs.dim() - 1, dtype=int))) + \
                          torch.randn_like(inputs) * work_alpha.view((-1, *np.ones(inputs.dim() - 1, dtype=int)))

        # зашумить на 1 шаг
        noise = torch.randn_like(denoise_targets)
        input_img = denoise_targets * np.sqrt(1. - alpha) + noise * np.sqrt(alpha)

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

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, inputs_targets.shape).to(
            inputs_targets.get_device())
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, inputs_targets.shape).to(
            inputs_targets.get_device())

        input_img = sqrt_alphas_cumprod_t * inputs_targets + sqrt_one_minus_alphas_cumprod_t * noise

        return input_img.requires_grad_(), noise

    return diff_prepare_data
    '''

    """

    def diffusionPrepareFun(self, data_package):
        raise NotImplemented(
            " diffusionPrepareFun FUN DONT IMPLEMENTED!")  #####################################################################################
        return (data_package[0], t), data_package[1], data_package[2] if self.use_count_pixel_balance else None

    return diffusionPrepareFun

def getPrepareDataFunction(task_mode, train_args=None):  # data_package: [input, target, statistic]
    if task_mode == "segmentation":
        return segmentationPrepareFun
    elif task_mode == "img2img":
        return img2imgPrepareFun
    elif task_mode == "diffusion":
        return diffusionPrepareFunWarper(train_args)
    else:
        msg = f"ERROR!!! Train mode don't know {task_mode}. Available optimizer options {', '.join(AVAILABLE_PREPARE_OPTIONS)}"
        raise AttributeError(msg)

#################### PREDICT MODEL BLOCK ############################

class tiledGen():
    def __init__(self, data, batch_size=1):
        self.batch_size = batch_size
        self.data = data
    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration

        items = self.data[index*self.batch_size : min((index + 1) * self.batch_size, len(self.data))]
        tensor_data = torch.stack([torch.from_numpy(item) for item in items])
        if len(tensor_data.shape) == 3:
            tensor_data = tensor_data.unsqueeze(-1)
        tensor_data = tensor_data.type(torch.FloatTensor).permute(0, 3, 1, 2)

        return tensor_data

    def __len__(self):
        return len(self.data) // self.batch_size + (0 if len(self.data) % self.batch_size == 0 else 1)

def prepare_list_batch_to_list_imgs(data):
    res_data = []
    for batch in data:
        for img in batch:
            res_data.append(img)
    return res_data

def saveBatchResultMask(save_path, npyfile, namelist, classnames=None):

    save_mask_dir = os.path.abspath(save_path)
    if os.name == 'nt':  # for Windows
        if save_mask_dir.startswith(u"\\\\"):
            save_mask_dir = u"\\\\?\\UNC\\" + save_mask_dir[2:]
        else:
            save_mask_dir = u"\\\\?\\" + save_mask_dir

    for i, item in enumerate(npyfile):
        num_class = item.shape[2]
        for class_index in range(num_class):
            out_dir = os.path.join(save_mask_dir,
                                   classnames[class_index] if classnames is not None else str(class_index))
            if not os.path.isdir(out_dir):
                print("создаю out_dir:" + out_dir.replace(u"\\\\?\\" + os.getcwd() + "\\", ""))
                os.makedirs(out_dir)

            #if (os.path.isfile(os.path.join(out_dir, "predict_" + namelist[i]))):
            #    os.remove(os.path.join(out_dir, "predict_" + namelist[i]))

            io.imsave(os.path.join(out_dir, "predict_" + namelist[i]), item[:, :, class_index],
                      check_contrast=False)

def saveResultMask(save_path, mask, name_file, classnames=None):

    save_mask_dir = os.path.abspath(save_path)
    if os.name == 'nt':  # for Windows
        if save_mask_dir.startswith(u"\\\\"):
            save_mask_dir = u"\\\\?\\UNC\\" + save_mask_dir[2:]
        else:
            save_mask_dir = u"\\\\?\\" + save_mask_dir

    num_class = mask.shape[2]
    for class_index in range(num_class):
        out_dir = os.path.join(save_mask_dir,
                               classnames[class_index] if classnames is not None else str(class_index))
        if not os.path.isdir(out_dir):
            print("создаю out_dir:" + out_dir.replace(u"\\\\?\\" + os.getcwd() + "\\", ""))
            os.makedirs(out_dir)

        #if (os.path.isfile(os.path.join(out_dir, "predict_" + namelist[i]))):
        #    os.remove(os.path.join(out_dir, "predict_" + namelist[i]))

        io.imsave(os.path.join(out_dir, "predict_" + name_file), to_0_255_format_img(mask[:, :, class_index]),
                  check_contrast=False)

def plug_old_connect_res_test_data(list_img, list_name):
    return list(zip(list_name, list_img))

def extract_channels_by_indexes_from_img_list(list_img, list_indexes):
    res_list = []
    for img in list_img:
        res_list.append(img.take(indices=list_indexes, axis=-1))
    return res_list

def saveImg(save_path, img):
    print("save", save_path)
    io.imsave(save_path, img, check_contrast=False)

def saveDataframeAsImgs(save_path, filename, dataframe):
    print("save dataframe", save_path)
    for z, img in enumerate(dataframe): # shape(d, h, w, c)
        io.imsave(os.path.join(save_path, filename + f"_{z}.png"), img, check_contrast=False)

