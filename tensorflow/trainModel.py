from model import *
from dataGenerator import *
from metrics import *
#import viewerLearningRate

import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import json

'''
#GPU desable
try:
    # Disable all GPUS
   tf.config.set_visible_devices([], 'GPU')
   visible_devices = tf.config.get_visible_devices()
   for device in visible_devices:
       assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
'''


if __name__ == '__main__':
    import tensorflow.compat.v1.keras.backend as KTF

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    KTF.set_session(sess)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = None) 
    args = parser.parse_args()
    return args

def scheduler(epoch, lr):
    if epoch < 100:
        return 0.0001
    elif epoch < 125:
        return 0.00005
    elif epoch < 150:
        return 0.00001
    elif epoch < 175:
        return 0.000005
    return 0.000001
   
class End_Fun(keras.callbacks.Callback):
    def __init__(self, Gen):
        self.Gen_end = Gen.on_epoch_end
    def on_epoch_end(self, epoch = '', epoch_data_val = '', tree = ''):
        #print("End_Fun epoch", epoch, "epoch_data_val ", two, "end", tree)
        self.Gen_end()

def config_parcer(dict_config):
    
    augmentation = dict_config["augmentation"]
    
    if not augmentation["rotation_range"]:
        augmentation["rotation_range"] = 0
    if not augmentation["width_shift_range"]:
        augmentation["width_shift_range"]  = 0
    if not augmentation["height_shift_range"]:
        augmentation["height_shift_range"] = 0     
    if not augmentation["zoom_range"]:
        augmentation["zoom_range"] = 0     
    if not augmentation["horizontal_flip"]:
        augmentation["horizontal_flip"] = False 
    if not augmentation["vertical_flip"]:
        augmentation["vertical_flip"] = False 
    if not augmentation["noise_limit"]:
        augmentation["noise_limit"] = 0     
    if not augmentation["fill_mode"]:
        augmentation["fill_mode"] = 0  
    
    
    dir_data = InfoDirData(dir_img_name = dict_config["train"]["dir_img_path"],
               dir_mask_name            = dict_config["train"]["dir_mask_path_without_name"],
               add_mask_prefix          = dict_config["data_info"]["add_mask_prefix"])
    
    transform_data = TransformData(color_mode_img = dict_config["img_transform_data"]["color_mode_img"],
                                   mode_mask      = dict_config["img_transform_data"]["mode_mask"],
                                   target_size    = dict_config["img_transform_data"]["target_size"],
                                   batch_size     = dict_config["train"]["batch_size"])

    save_inform = SaveData(save_to_dir       = dict_config["save_inform"]["save_to_dir"],
                           save_prefix_image = dict_config["save_inform"]["save_prefix_image"],
                           save_prefix_mask  = dict_config["save_inform"]["save_prefix_mask"])

    if not dict_config["generator_config"]["type_gen"] or dict_config["generator_config"]["type_gen"] == "default":
        myGen = DataGenerator(dir_data = dir_data,
                          num_classes     = dict_config["train"]["num_class"],
                          mode            = dict_config["generator_config"]["mode"],
                          aug_dict        = augmentation,
                          list_class_name = dict_config["train"]["mask_name_label_list"],
                          augment         = dict_config["generator_config"]["augment"],
                          tailing         = dict_config["generator_config"]["tailing"],
                          shuffle         = dict_config["generator_config"]["shuffle"],
                          seed            = dict_config["generator_config"]["seed"],
                          subsampling     = dict_config["generator_config"]["subsampling"],
                          transform_data  = transform_data,
                          save_inform     = save_inform,
                          share_validat   = dict_config["generator_config"]["share_validat"])
    elif dict_config["generator_config"]["type_gen"] == "all_reader":
            myGen = DataGeneratorReaderAll(dir_data = dir_data,
                          num_classes     = dict_config["train"]["num_class"],
                          mode            = dict_config["generator_config"]["mode"],
                          aug_dict        = augmentation,
                          list_class_name = dict_config["train"]["mask_name_label_list"],
                          augment         = dict_config["generator_config"]["augment"],
                          tailing         = dict_config["generator_config"]["tailing"],
                          shuffle         = dict_config["generator_config"]["shuffle"],
                          seed            = dict_config["generator_config"]["seed"],
                          subsampling     = dict_config["generator_config"]["subsampling"],
                          transform_data  = transform_data,
                          save_inform     = save_inform,
                          share_validat   = dict_config["generator_config"]["share_validat"])
    else:
        print("GEN CHOISE ERROR: now you can only choose: 'default', 'all_reader' generator")
        raise AttributeError("GEN CHOISE ERROR: now you can only choose: 'default', 'all_reader' generator")
    
    if not dict_config["model"]["optimizer"] or dict_config["model"]["optimizer"] == "Adam":
        if not dict_config["model"]["optimizer"]:
            dict_config["model"]["optimizer"] = "Adam"
        optimizer = tf.optimizers.Adam(learning_rate = 1e-4)
    elif dict_config["model"]["optimizer"] == "AdamW":
        optimizer=tfa.optimizers.AdamW(learning_rate = 1e-4, weight_decay = 1e-4)
    elif dict_config["model"]["optimizer"] == "RMSprop":
        optimizer=tf.optimizers.RMSprop(learning_rate = 5e-3)
    elif dict_config["model"]["optimizer"] == "NovoGrad":
        optimizer=tfa.optimizers.NovoGrad(learning_rate= 1e-3, weight_decay = 1e-3)
    else:
        print("OPTIMIZER CHOISE ERROR: now you can only choose: 'Adam', 'AdamW', 'RMSprop', 'NovoGrad' optimizer")
        raise AttributeError("OPTIMIZER CHOISE ERROR: now you can only choose: 'Adam', 'AdamW', 'RMSprop', 'NovoGrad' optimizer")
  
    num_channel = 1 if dict_config["img_transform_data"]["color_mode_img"] == 'gray' else 3
    
    if not dict_config["model"]["type_model"] or dict_config["model"]["type_model"] == "unet":
        if not dict_config["model"]["type_model"]:
            dict_config["model"]["type_model"] = "unet"
        model = unet(num_class = dict_config["train"]["num_class"], input_size= (*(dict_config["img_transform_data"]["target_size"]), num_channel))
    elif dict_config["model"]["type_model"] == "tiny_unet":
        model = tiny_unet(num_class = dict_config["train"]["num_class"], input_size= (*(dict_config["img_transform_data"]["target_size"]), num_channel))
    elif  dict_config["model"]["type_model"] == "tiny_unet_v3":
        model = tiny_unet_v3(num_class = dict_config["train"]["num_class"], input_size= (*(dict_config["img_transform_data"]["target_size"]), num_channel))
    elif dict_config["model"]["type_model"] == "mobile_unet_v2":
        model = mobile_unet_v2(num_class = dict_config["train"]["num_class"], input_size= (*(dict_config["img_transform_data"]["target_size"]), num_channel))
    else:
        print("MODEL CHOISE ERROR: now you can only choose: 'unet', 'tiny_unet', 'tiny_unet_v3', 'mobile_unet_v2' model")
        raise AttributeError("MODEL CHOISE ERROR: now you can only choose: 'unet', 'tiny_unet', 'tiny_unet_v3', 'mobile_unet_v2' model")


    loss = []
    if not "loss" in dict_config["model"] or dict_config["model"]["loss"] ==  "dice_distance" or "dice_distance" in dict_config["model"]["loss"]:
        if not "loss" in dict_config["model"]:
            dict_config["model"]["loss"] = "dice_distance"
        loss.append(universal_dice_coef_loss(dict_config["train"]["num_class"]))
    if dict_config["model"]["loss"] == "BCE" or "BCE" in dict_config["model"]["loss"]:
        loss.append(balanced_cross_entropy(0.7))
 
    if len(loss) == 0:
        print("LOSS CHOISE ERROR: now you can only choose: 'dice_distance', 'BCE' loss")
        raise AttributeError("LOSS CHOISE ERROR: now you can only choose: 'dice_distance', 'BCE' loss")

    print("losses:", loss)

    if "pretrained_weights" in dict_config["train"]:
        pretrained_weights = dict_config["train"]["pretrained_weights"]
    else:
        pretrained_weights = None
    # model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[universal_dice_coef_multilabel(dict_config["train"]["num_class"]), universal_dice_coef_multilabel_arr(dict_config["train"]["num_class"])])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    #print (model.summary())    
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes=True)

    num_epochs = dict_config["train"]["num_epochs"]

    modelName = ""

    if len(dict_config["save_inform"]["save_prefix_model"]) > 0:
        modelName += dict_config["save_inform"]["save_prefix_model"] + "_"

                 #str(dict_config["img_transform_data"]["target_size"][0]) + "_" +\
    modelName += dict_config["model"]["type_model"] #+ "_"  +\
    # str(dict_config["train"]["num_class"]) + "_num_class" #+ "_" +\
                 #dict_config["model"]["optimizer"] 
      
    if len(dict_config["save_inform"]["save_suffix_model"]) > 0:
        modelName += "_" + dict_config["save_inform"]["save_suffix_model"]
     
    if dict_config["debug_mode"]:
        print ("print myGen:")
        print ("\ttypeGen:", myGen.typeGen)
        print ("\tdir_data:", "dir_img_name", myGen.dir_data.dir_img_name, ", dir_mask_name", myGen.dir_data.dir_mask_name, ", add_mask_prefix",  myGen.dir_data.add_mask_prefix)
        print ("\tlist_class_name:", myGen.list_class_name)
        print ("\tnum_classes:", myGen.num_classes)
        print ("\ttransform_data:", ", color_mode_img",  myGen.transform_data.color_mode_img, ", mode_mask", myGen.transform_data.mode_mask, ", target_size", myGen.transform_data.target_size, ", batch_size",  myGen.transform_data.batch_size)
        print ("\taug_dict:", myGen.aug_dict)
        print ("\tmode:", myGen.mode)
        print ("\tsubsampling:", myGen.subsampling)
        print ("\tsave_inform:", " save_to_dir ", myGen.save_inform.save_to_dir, " save_prefix_image ", myGen.save_inform.save_prefix_image, " save_prefix_mask ", myGen.save_inform.save_prefix_mask)
        print ("\tshare_validat:", myGen.share_val)
        print ("\taugment:", myGen.augment)
        print ("\tshuffle:", myGen.shuffle)
        print ("\tseed:", myGen.seed)
        print ("\ttailing:", myGen.tailing)
        print ("\tlen list_img_name:", len(myGen.list_img_name))
        print()
        
        print ("print Model:")
        print ("\tnum_epochs:", num_epochs)
        print ("\tmodelName:", modelName)
        print ("\toptimizer:", optimizer)
        print ("\tModelSize:", model.count_params())
        print()

    return  myGen, model, num_epochs, modelName
    
callback = LearningRateScheduler(scheduler, verbose=1)

import setproctitle

def train_by_config(config_file, path_config = None):

    data_save = None
        
    if config_file is None:
        myGen = DataGenerator()
        model = unet()
        model.compile(optimizer="adam",
                  loss=[universal_dice_coef_loss(2)],
                  metrics=[universal_dice_coef_multilabel(2)])
        epochs=200
        modelName = "no_config_unet_model"
    else:
        setproctitle.setproctitle(os.path.basename(path_config)[:-5])
        myGen, model, epochs, modelName = config_parcer(config_file)
        
        modelName = "model_by_" + os.path.basename(path_config)[:-5] + "_" + modelName
        
        if (config_file["move_to_date_folder"]):
            data_save = config_file["experiment_data"]
    
    ##################################################################################################################
    ######### main train function ##########################
    end_epoch = End_Fun(myGen)

    model_checkpoint = ModelCheckpoint(modelName + '.hdf5', mode='min', monitor='val_loss',verbose=1, save_best_only=True)
    history = model.fit(myGen.gen_train, epochs=epochs, callbacks=[model_checkpoint, callback, end_epoch], workers = 3, verbose=1, validation_data=myGen.gen_valid)
    ##################################################################################################################

    print ("Finish Train")

    # можно заменить историю на добавление данных в End_Fun и смотреть тренировку в реальном времени.
    try:
        history.history['lr'] = np.array(history.history['lr']).astype(float).tolist()
    except:
        print("WARNING: no lr")
    #print(history.history)
    #save history
    import json
    with open("history_" + modelName + '.json', 'w') as file:
        json.dump(history.history, file, indent=4)

    if config_file is not None and (config_file["move_to_date_folder"]):
        import shutil

        if not os.path.isdir(data_save):
            print("create dir:", data_save)
            os.mkdir(data_save)
        print("move model, history and config in :", data_save)
    
        with open(os.path.join(data_save, os.path.basename(path_config)), 'w') as file:
            json.dump(config_file, file, indent=4)
        shutil.move(modelName + '.hdf5', os.path.join(data_save, modelName + '.hdf5'))
        shutil.move("history_" + modelName + '.json', os.path.join(data_save, "history_" + modelName + '.json'))

if __name__ == '__main__':
    args = build_argparser()
    
    if args.config:
        path = args.config
        with open(args.config) as config_buffer:    
                config = json.loads(config_buffer.read())
    else:
        config = None
        path = None
        
    train_by_config(config, path)
     
    #viewerLearningRate.viewData(history.history) 
