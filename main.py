from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#чтобы избавиться от 0 (нулевого) лишнего класса
#нужно переделать разбивку данных маски при загрузки и избавится тем самым от 0 класса
num_class = 6

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1,'data/myltidata/train4','image','mask',data_gen_args, flag_multi_class = True, num_class = num_class, save_to_dir = None)#'data/myltidata/train3/temp')

#model = unet(num_class = num_class)
model = unet("unet_myltidata_v3.hdf5", num_class = num_class)

model_checkpoint = ModelCheckpoint('unet_myltidata_v3.hdf5', mode='auto', monitor='loss',verbose=1, save_best_only=True)

model.fit_generator(myGene, steps_per_epoch=100, epochs=10,callbacks=[model_checkpoint], verbose=1)


#saveResult("data/membrane/test",results)

name_list = []
#testGene = testGenerator("data/myltidata/test/img", size_test_train, flag_multi_class = True)
testGene = testGenerator2("data/myltidata/test/splitted",  name_list, 12, flag_multi_class = True)

results = model.predict_generator(testGene, 12, verbose=1)

#saveResult("data/myltidata/result3", results, trust_percentage = 0.85, flag_multi_class = True, num_class = num_class)
saveResult2("data/myltidata/splitted", results, name_list, trust_percentage = 0.95, flag_multi_class = True, num_class = num_class)