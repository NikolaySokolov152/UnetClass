# этот скрипт запускает последовательно обучение одной конфигурации с разным количеством классов

from trainModel import *
import json

import tensorflow.compat.v1.keras.backend as KTF

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

KTF.set_session(sess)

classes = [6, 5, 1]

if __name__ == '__main__':
    args = build_argparser()
    
    if args.config:
        path = args.config
        with open(args.config) as config_buffer:    
            config = json.loads(config_buffer.read())
    else:
        print("ERROR OPEN CONFIG")
        raise("ERROR OPEN CONFIG")
         
    change_save_suffix = config["save_inform"]["save_suffix_model"]
    #config["move_to_date_folder"] = False
 
    for n_class in classes:
        print()
        print("Learning ", n_class, "classes model", config["model"]["type_model"])
        print()
        
    
        #config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config["train"]["num_class"] = n_class
        train_by_config(config, os.path.basename(path)[:-5] + "_" + str(n_class) + "_classes.json")
    