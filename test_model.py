from model import *
from data import *

#rgb
#смена цвета тут не на что не влияет, просто тут удобнее смотреть цвет
#для смены цвета менять в data
any                 = [192, 192, 192]   #wtite-gray
borders             = [0,0,255]         #blue
mitochondria        = [0,255,0]         #green
mitochondria_borders= [255,0,255]       #violet
PSD                 = [192,192,64]      #yellow
vesicles            = [255,0,0]         #read

COLOR_DICT = np.array([any, mitochondria, mitochondria_borders, PSD, vesicles,borders])

num_class = 6

model = unet("unet_myltidata_v3.hdf5", num_class = num_class)

size_test_train = 60

name_list = []
#testGene = testGenerator("data/myltidata/test/img", size_test_train, flag_multi_class = True)
testGene = testGenerator2("data/myltidata/test/splitted",  name_list, size_test_train, flag_multi_class = True)

results = model.predict_generator(testGene, size_test_train, verbose=1)

#saveResult("data/myltidata/result3", results, trust_percentage = 0.85, flag_multi_class = True, num_class = num_class)
saveResult2("data/myltidata/splitted", results, name_list, trust_percentage = 0.85, flag_multi_class = True, num_class = num_class)

