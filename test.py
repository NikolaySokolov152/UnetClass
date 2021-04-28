from model import *
from data import *

#rgb
any                 = [192, 192, 192]   #wtite-gray
borders             = [0,0,255]         #blue
mitochondria        = [0,255,0]         #green
mitochondria_borders= [255,0,255]       #violet
PSD                 = [192,192,64]      #yellow
axon                = [192,128,64]      #yellow
vesicles            = [255,0,0]         #read

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


def test_modul_v5():
    num_class = 5

    model = unet('my_unet_multidata.hdf5', num_class = num_class)

    size_test_train = 4#60

    name_list = []
    testGene = testGenerator(test_path="data/test", name_list=name_list,\
                             #save_dir="data/result",\
                             num_image=size_test_train, flag_multi_class=True)

    results = model.predict(testGene, size_test_train, verbose=1)

    viewResult("data/result", results, name_list, trust_percentage=0.95, flag_multi_class=True, num_class=num_class)
    saveResult("data/result", results, name_list, trust_percentage = 0.95, flag_multi_class = True, num_class = num_class)

def test_modul_v6():
    num_class = 6

    model = unet('my_unet_multidata.hdf5', num_class = num_class)

    size_test_train = 60

    name_list = []
    testGene = testGenerator(test_path="data/test", name_list=name_list, save_dir="data/result",\
                             num_image=size_test_train, flag_multi_class=True)

    results = model.predict_generator(testGene, size_test_train, verbose=1)

    saveResult("data/result", results, name_list, trust_percentage = 0.95, flag_multi_class = True, num_class = num_class)

test_modul_v5()
#test_modul_v6()