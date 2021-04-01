from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#print(tf.test.is_built_with_cuda())
#print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))

#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)



num_class = 5 #6

data_gen_args = dict(rotation_range= 5,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')

mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boarders", "mitochondria borders"]

myGene = get_train_generator_data(dir_img_name = 'data/train/origin',
                                  dir_mask_name = 'data/train/',
                                  aug_dict = data_gen_args,
                                  batch_size = 2,
                                  list_name_label_mask = mask_name_label_list,
                                  delete_mask_name = None,
                                  target_size = (256,256),
                                  color_mode_img = "gray",
                                  color_mode_mask = "gray",
                                  normalase_img_mod = "div255",
                                  num_class = num_class,
                                  label_mask = False,
                                  normalase_mask_mode = "to_0_1", #"to_-1_1"
                                  save_prefix_image="image_",
                                  save_prefix_mask="mask_",
                                  save_to_dir = None, #"data/myltidata/train4/temp",
                                  seed = 1
                                  )

model = unet(num_class = num_class)
#model = unet('my_unet_data.hdf5', num_class = num_class)

model_checkpoint = ModelCheckpoint('my_unet_multidata.hdf5', mode='auto', monitor='loss',verbose=1, save_best_only=True)

model.fit_generator(myGene, steps_per_epoch=100, epochs=100,callbacks=[model_checkpoint], verbose=1 , validation_data=myGene, validation_steps=5)

name_list = []
testGene = testGenerator(test_path = "data/test", name_list = name_list, save_dir= "data/result", num_image = 12, flag_multi_class = True)

results = model.predict_generator(testGene, 12, verbose=1)

saveResult("data/result", results, name_list, trust_percentage = 0.8, flag_multi_class = True, num_class = num_class)