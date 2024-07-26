from dataGenerator import *
from models import *
from metrics import *

import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_0_255_format_img(in_img):
    max_val = in_img[:, :].max()
    if max_val <= 1:
        out_img = np.round(in_img * 255)
        return out_img.astype(np.uint8)
    else:
        return in_img

def to_numpy_from_torch(tensor):
    return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()

def printGenerator(gen, isPrintInput = True, isPrintLen = True):
    if isPrintInput:
        print("DEBUG_INPUT")
        print(gen.__class__)
        print('\tinform_generator:', gen.inform_generator)
        print('\tdir_data:', gen.dir_data)
        print('\tlist_class_name:', gen.list_class_name)
        print('\tnum_classes:', gen.num_classes)
        print('\ttransform_data:', gen.transform_data)
        print('\tmode:', gen.mode)
        print('\tsave_inform:', gen.save_inform)
        print('\ttailing:', gen.tailing)
        print('\taugment:', gen.augment)
        print('\tbatch_size:', gen.batch_size)
        print('\tsmall_list_img_name:', gen.small_list_img_name)
        print('\taug_dict:', gen.composition)
    if isPrintLen:
        print('DEBUG_LEN_GEN')
        print(gen.__class__)
        print(f'\tlen_gen: {len(gen)}')

def printDataGenerator(gen, isPrintInput = True, isPrintLen = True):
    if isPrintInput:
        print("DEBUG_INPUT")
        print(gen.__class__)
        print('\ttypeGen:', gen.typeGen)
        print('\tdir_data:', gen.dir_data)
        print('\tlist_class_name:', gen.list_class_name)
        print('\tnum_classes:', gen.num_classes)
        print('\ttransform_data:', gen.transform_data)
        print('\taug_dict:', gen.aug_dict)
        print('\tmode:', gen.mode)
        print('\tsubsampling:', gen.subsampling)
        print('\tsave_inform:', gen.save_inform)
        print('\tshare_val:', gen.share_val)
        print('\taugment:', gen.augment)
        print('\tshuffle:', gen.shuffle)
        print('\tseed:', gen.seed)
        print('\ttailing:', gen.tailing)
        print('\tlist_img_name:', gen.list_img_name)
    if isPrintLen:
        print('DEBUG_LEN_GEN')
        print(gen.__class__)
        print(f'\tlen_gen: {len(gen)}')

def printAugmentGenerator(gen, isPrintInput = True, isPrintLen = True):
    if isPrintInput:
        print("DEBUG_INPUT")
        print(gen.__class__)
        print('\tinform_generator:', gen.inform_generator)
        print('\ttransform_data:', gen.transform_data)
        print('\tmode:', gen.mode)
        print('\tsave_inform:', gen.save_inform)
        print('\ttailing:', gen.tailing)
        print('\taugment:', gen.augment)
        print('\tbatch_size:', gen.batch_size)
        print('\tlen_img:', len(gen.images))
        print('\tlen_mask:', len(gen.masks))
        print('\tindexes:', gen.indexes)
        print('\taug_dict:', gen.composition)
    if isPrintLen:
        print('DEBUG_LEN_GEN')
        print(gen.__class__)
        print(f'\tlen_gen: {len(gen)}')

def printDataGeneratorReaderAll(gen, isPrintInput = True, isPrintLen = True):
    if isPrintInput:
        print("DEBUG_INPUT")
        print(gen.__class__)
        print('\ttypeGen:', gen.typeGen)
        print('\tdir_data:', gen.dir_data)
        print('\tlist_class_name:', gen.list_class_name)
        print('\tnum_classes:', gen.num_classes)
        print('\ttransform_data:', gen.transform_data)
        print('\taug_dict:', gen.aug_dict)
        print('\tmode:', gen.mode)
        print('\tsubsampling:', gen.subsampling)
        print('\tsave_inform:', gen.save_inform)
        print('\tshare_val:', gen.share_val)
        print('\taugment:', gen.augment)
        print('\tshuffle:', gen.shuffle)
        print('\tseed:', gen.seed)
        print('\ttailing:', gen.tailing)
        print('\tlist_img_name:', gen.list_img_name)
        print('\tindexes:', gen.indexes)

    if isPrintLen:
        print('DEBUG_LEN_GEN')
        print(gen.__class__)
        print(f'\tlen_gen: {len(gen)}')

def viewDataBatchCV(x,y):
    for i in range(x.shape[0]):
        # print(str("x")+":",  x[i].max(), " ",x[i].min())
        # print(str("y")+":",  y[i].max(), " ",y[i].min())
        num_class = y.shape[-1]

        cv2.imshow(f"X batch {i}", x[i])

        for k in range(num_class):
            cv2.imshow(f"Y batch {i}, class {k}", to_0_255_format_img(y[i, :, :, k]))

    cv2.waitKey()

def viewDataBatchPlot(x, y):
    counter = 1

    fig = plt.figure(figsize=(20, 12), dpi=100)
    for i in range(x.shape[0]):
        # print(str("x")+":",  x[i].max(), " ",x[i].min())
        # print(str("y")+":",  y[i].max(), " ",y[i].min())

        num_class = y.shape[-1]

        fig.suptitle("view batch", fontsize=16)

        ax = fig.add_subplot(x.shape[0], y.shape[-1]+1, counter)
        ax.imshow(x[i])
        counter += 1

        for k in range(num_class):
            ax = fig.add_subplot(x.shape[0], y.shape[-1]+1, counter)
            ax.get_yaxis().set_visible(False)
            ax.imshow( to_0_255_format_img(y[i, :, :, k]))

            counter += 1

        fig.tight_layout()
    plt.show()

def viewDataBatch(x, y, isFast = True):
    if isFast:
        viewDataBatchCV(x, y)
    else:
        viewDataBatchPlot(x, y)

def weak_test_gen_DataGeneratorReaderAll():

    num_class = 2

    data_gen_args = dict(zoom_range=0.2,
                         brightness_shift_range=0.2,
                         contrast_shift_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         noise_limit=3,
                         rotate_90=.5,
                         fill_mode=0)  # cv2.BORDER_CONSTANT = 0


    # берутся первые классы из списка
    mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]
    #mask_name_label_list = ["output"]

    dir_data = [InfoDirData(dir_img_name="D:/Projects/Synthetics/dataset/synthetic_dataset10/original",
                           dir_mask_name="D:/Projects/Synthetics/dataset/synthetic_dataset10/",
                           add_mask_prefix='',
                           proportion_of_dataset=0.2),

               InfoDirData(dir_img_name = "D:/Data/Unet_multiclass/data/cutting data/original",
                           dir_mask_name ="D:/Data/Unet_multiclass/data/cutting data/",
                           add_mask_prefix = '',
                           proportion_of_dataset=0.2)]

    dir_data = [dir_data[1], dir_data[0]]

    #dir_data = InfoDirData(dir_img_name="data/zip_data_train/original.npy",
    #                       dir_mask_name="data/zip_data_train/mask.npy",
    #                       add_mask_prefix='')

    dir_data = InfoDirData(common_dir_path="D:/Data/Unet_multiclass/data/original data/",
                           dir_img_name="original",
                           add_mask_prefix='')


    transform_data = TransformData(color_mode_img='gray',
                                   mode_mask='separated',
                                   target_size=(256, 256),
                                   batch_size=3)

    save_inform = SaveData(save_to_dir=None,
                           save_prefix_image="image_",
                           save_prefix_mask="mask_")

    # try:
    # myGen = DataGenerator(dir_data = dir_data,
    myGen = DataGeneratorReaderAll(dir_data=dir_data,
                                   num_classes=num_class,
                                   mode="train",
                                   aug_dict=data_gen_args,
                                   list_class_name=mask_name_label_list,
                                   augment=True,
                                   tailing=True,
                                   shuffle=True,
                                   type_load_data="img",
                                   seed=1,
                                   num_gen_repetitions=10,
                                   subsampling="random",
                                   transform_data=transform_data,
                                   save_inform=save_inform,
                                   share_validat=0.2)
    # except Exception as e: print(e)

    printDataGeneratorReaderAll(myGen)
    printAugmentGenerator(myGen.gen_train)
    printAugmentGenerator(myGen.gen_valid)

    #myGen.saveNpyData()


    count = 0

    print(len(myGen))
    print(len(myGen.gen_train))

    print(len(myGen.gen_valid))
    size_train = len(myGen.gen_train)

    #gen_train = myGen.gen_train
    gen_train = myGen

    print(type(myGen.gen_train))

    print("\ntrain\n")

    #for x, y in (gen_train):
    for x,y in tqdm(myGen.gen_train,
                   desc="\t",
                   file=sys.stdout,
                   colour="GREEN",
                   disable=False):
        x = to_numpy_from_torch(x)
        y = to_numpy_from_torch(y)

        y_no_1 = y[y!=1.0]
        y_no_01 = y_no_1[y_no_1!=0.0]

        if len(y_no_01):
            print("ERROR! mask have no binary data", len(y_no_01), y_no_01)

        #print(x.shape, " ", y.shape)

        viewDataBatch(x, y)

        count += 1

    print("\nvalid\n")

    for i in range(2 * len(myGen.gen_valid)):
        x, y = myGen.gen_valid[i]
        x = to_numpy_from_torch(x)
        y = to_numpy_from_torch(y)
        print(x.shape, " ", y.shape)

        viewDataBatch(x, y)
        count += 1

    myGen.on_epoch_end()

    print("\ntrain\n")

    for i in range(2 * size_train):
        x, y = gen_train[i]
        x = to_numpy_from_torch(x)
        y = to_numpy_from_torch(y)
        print(x.shape, " ", y.shape)


        viewDataBatch(x, y)
        count += 1

    print("\nvalid\n")
    for i in range(2 * len(myGen.gen_valid)):
        x, y = myGen.gen_valid[i]
        x = to_numpy_from_torch(x)
        y = to_numpy_from_torch(y)
        print(x.shape, " ", y.shape)


        viewDataBatch(x, y)
        count += 1

    myGen.on_epoch_end()
    myGen.on_epoch_end()
    myGen.on_epoch_end()
    myGen.on_epoch_end()

def weak_test_gen_DataGenerator():
    num_class = 1

    data_gen_args = dict(rotation_range=15,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         noise_limit=5,
                         fill_mode=0)  # cv2.BORDER_CONSTANT = 0

    # берутся первые классы из списка
    mask_name_label_list = ["output"]

   # dir_data = InfoDirData(dir_img_name="C:/Users/Sokol-PC/Synthetics/dataset/fake_synthetic_layer/original",
   #                        dir_mask_name="C:/Users/Sokol-PC/Synthetics/dataset/fake_synthetic_layer/",
    #                       add_mask_prefix='')

    #dir_data = InfoDirData(dir_img_name="C:/Users/Sokol-PC/bachelor_diploma/train/original",
    #                       dir_mask_name="C:/Users/Sokol-PC/bachelor_diploma/train/",
    #                       add_mask_prefix='')

    # dir_data = InfoDirData(dir_img_name = "G:/Data/Unet_multiclass/data/cutting data/original/",
    #                       dir_mask_name = "G:/Data/Unet_multiclass/data/cutting data/",
    #                       add_mask_prefix = '')

    dir_data = InfoDirData(dir_img_name="D:/Projects/UnetClass/pytorch/unet_standart/img2img/data/cutting data/low_quality",
                           dir_mask_name="D:/Projects/UnetClass/pytorch/unet_standart/img2img/data/cutting data/",
                           add_mask_prefix='')

    transform_data = TransformData(color_mode_img='gray',
                                   mode_mask='image',
                                   target_size=(256, 256),
                                   batch_size=3)

    save_inform = SaveData(save_to_dir=None,
                           save_prefix_image="image_",
                           save_prefix_mask="mask_")

    # try:
    # myGen = DataGenerator(dir_data = dir_data,
    myGen = DataGenerator(dir_data=dir_data,
                                   num_classes=num_class,
                                   mode="train",
                                   aug_dict=data_gen_args,
                                   list_class_name=mask_name_label_list,
                                   augment=True,
                                   tailing=False,
                                   shuffle=True,
                                   seed=1,
                                   subsampling="random",
                                   transform_data=transform_data,
                                   save_inform=save_inform,
                                   share_validat=0.2)
    # except Exception as e: print(e)

    printDataGenerator(myGen)
    printGenerator(myGen.gen_train)
    printGenerator(myGen.gen_valid)

    count = 0

    print(len(myGen))
    print(len(myGen.gen_train))

    print(len(myGen.gen_valid))
    size_train = len(myGen.gen_train)

    gen_train = myGen.gen_train

    print(type(myGen.gen_train))

    print("\ntrain\n")

    for i in range(2 * size_train):
        x, y = gen_train[i]
        x = to_numpy_from_torch(x)
        y = to_numpy_from_torch(y)
        print(x.shape, " ", y.shape)

        viewDataBatch(x, y)

        count += 1

    print("\nvalid\n")

    for i in range(2 * len(myGen.gen_valid)):
        x, y = myGen.gen_valid[i]
        x = to_numpy_from_torch(x)
        y = to_numpy_from_torch(y)
        print(x.shape, " ", y.shape)

        viewDataBatch(x, y)
        count += 1

    myGen.on_epoch_end()

    print("\ntrain\n")

    for i in range(2 * size_train):
        x, y = gen_train[i]
        x = to_numpy_from_torch(x)
        y = to_numpy_from_torch(y)
        print(x.shape, " ", y.shape)


        viewDataBatch(x, y)
        count += 1

    print("\nvalid\n")
    for i in range(2 * len(myGen.gen_valid)):
        x, y = myGen.gen_valid[i]
        x = to_numpy_from_torch(x)
        y = to_numpy_from_torch(y)
        print(x.shape, " ", y.shape)


        viewDataBatch(x, y)
        count += 1

    myGen.on_epoch_end()
    myGen.on_epoch_end()
    myGen.on_epoch_end()
    myGen.on_epoch_end()

weak_test_gen_DataGeneratorReaderAll()
#weak_test_gen_DataGenerator()