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

    model = unet('my_unet_multidata_pe38_bs7_5class.hdf5', num_class = num_class)

    size_test_train = 12

    name_list = []
    testGene = testGenerator(test_path="data/test", name_list=name_list,\
                             #save_dir="data/result",\
                             num_image=size_test_train, flag_multi_class=True)

    results = model.predict(testGene, size_test_train, verbose=1)

    #viewResult("data/result", results, name_list, trust_percentage=0.95, flag_multi_class=True, num_class=num_class)
    saveResultMask("data/result", results, name_list, num_class = num_class)
    #saveResult("data/result", results, name_list, trust_percentage = 0.99, flag_multi_class = True, num_class = num_class)

def test_modul_v6():
    num_class = 6

    model = unet('my_unet_multidata_pe38_bs7_6class.hdf5', num_class = num_class)

    size_test_train = 12

    name_list = []
    testGene = testGenerator(test_path="data/test", name_list=name_list, save_dir="data/result",\
                             num_image=size_test_train, flag_multi_class=True)

    results = model.predict_generator(testGene, size_test_train, verbose=1)
    saveResultMask("data/result", results, name_list, num_class=num_class)
    #saveResult("data/result", results, name_list, trust_percentage = 0.95, flag_multi_class = True, num_class = num_class)

def test_modul_v5_one_pic():
    num_class = 5
    model = unet('my_unet_multidata_pe38_bs7_5class.hdf5', num_class = num_class)

    img_name = "training0003.png"
    img = io.imread(os.path.join("data/test", img_name), as_gray=True)
    #img = img / 255
    img = trans.resize(img, (256,256))
    img = np.reshape(img, (1,) + img.shape)
    io.imsave(os.path.join("data/result", img_name), img[0])

    results = model.predict(x=img, batch_size=1, verbose=1)

    results = [trans.resize(results[0], (768,1024,num_class))]

    saveResultMask("data/result", results, [img_name], num_class=num_class)
    #saveResult("data/result", results, name_list, trust_percentage = 0.95, flag_multi_class = True, num_class = num_class)

def test_modul_v6_one_pic():
    num_class = 6
    model = unet('my_unet_multidata_pe38_bs7_6class.hdf5', num_class = num_class)

    img_name = "training0003.png"
    img = io.imread(os.path.join("data/test", img_name), as_gray=True)
    #img = img / 255
    img = trans.resize(img, (256,256))
    img = np.reshape(img, (1,) + img.shape)
    io.imsave(os.path.join("data/result", img_name), img[0])

    results = model.predict(x=img, batch_size=1, verbose=1)

    results = [trans.resize(results[0], (768,1024,num_class))]

    saveResultMask("data/result", results, [img_name], num_class=num_class)
    #saveResult("data/result", results, name_list, trust_percentage = 0.95, flag_multi_class = True, num_class = num_class)

def test_modul_v1():
    num_class = 1

    model = unet('my_unet_multidata_pe38_bs7_1class.hdf5', num_class = num_class)

    size_test_train = 12

    name_list = []
    testGene = testGenerator(test_path="data/test", name_list=name_list, save_dir="data/result",\
                             num_image=size_test_train, flag_multi_class=True)

    results = model.predict_generator(testGene, size_test_train, verbose=1)
    saveResultMask("data/result", results, name_list, num_class=num_class)
    #saveResult("data/result", results, name_list, trust_percentage = 0.95, flag_multi_class = True, num_class = num_class)

def test_modul_v1_one_pic():
    num_class = 1
    model = unet('my_unet_multidata_pe38_bs7_1class.hdf5', num_class = num_class)

    img_name = "training0003.png"
    img = io.imread(os.path.join("data/test", img_name), as_gray=True)
    #img = img / 255
    img = trans.resize(img, (256,256))
    img = np.reshape(img, (1,) + img.shape)
    io.imsave(os.path.join("data/result", img_name), img[0])

    results = model.predict(x=img, batch_size=1, verbose=1)

    results = [trans.resize(results[0], (768,1024,num_class))]

    saveResultMask("data/result", results, [img_name], num_class=num_class)
    #saveResult("data/result", results, name_list, trust_percentage = 0.95, flag_multi_class = True, num_class = num_class)



#test_modul_v1()
#test_modul_v5()
#test_modul_v6()


#test_modul_v1_one_pic()
#test_modul_v5_one_pic()
test_modul_v6_one_pic()