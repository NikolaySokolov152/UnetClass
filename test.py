from model import *
from data import *

import json

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

def to_0_255_format_img(in_img):
    max_val = in_img[:,:].max()
    if max_val <= 1:
       out_img = np.round(in_img * 255)
       return out_img.astype(np.uint8)
    else:
        return in_img

def to_0_1_format_img(in_img):
    max_val = in_img[:,:].max()
    if max_val <= 1:
        return in_img
    else:
        out_img = in_img / 255
        return out_img


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




#old version
def overlay_splitting(img, size_data, size_step):
    h, w = img.shape[0:2]
    cutting_img_arr = []
    count_x_y = 0
    count_y = 0
    for start_y in range(0, h, size_step):
        if (h - start_y < size_data):
            continue
        count_y += 1
        for start_x in range(0, w, size_step):
            if (w - start_x < size_data):
                continue
            count_x_y += 1
            cutting_img_arr.append(img[start_y:start_y + size_data, start_x:start_x + size_data])

    if (count_y != 0):
        count_x = count_x_y // count_y
    else:
        print("error split")
        count_x = 0
    return (cutting_img_arr, ((count_x, count_y), size_data, size_step))

def overlay_splitting_with_superimposition(img, size_data, size_step, minimum_overlay_size):
    h, w = img.shape[0:2]
    cutting_img_arr = []
    overlay_coord_arr = []
    count_x_y = 0
    count_y = 0

    flag_error_step_y = False
    end_y = 0
    for y in range(0, h, size_step):
        if flag_error_step_y:
            break
        if (h - y <= size_data):
            #print("h - end_y", h - end_y)
            if (h - end_y < minimum_overlay_size):
                #print("continue y")
                break
            flag_error_step_y = True

        count_y += 1
        last_end_y = end_y

        if flag_error_step_y:
            start_y = h - size_data
            end_y = h
        else:
            start_y = y
            end_y = y + size_data

        if y != 0:
            overlay_coord_arr.append((last_end_y - start_y, 0))

        flag_error_step_x = False
        end_x = 0
        for x in range(0, w, size_step):
            if flag_error_step_x:
                break

            if (w - x <= size_data):
                if (w - end_x < minimum_overlay_size):
                    #print("continue x")
                    break
                flag_error_step_x = True

            count_x_y += 1
            last_end_x = end_x

            if flag_error_step_x:
                start_x = w - size_data
                end_x = w
            else:
                start_x = x
                end_x = x + size_data

            if (x != 0):
                overlay_coord_arr.append((0, last_end_x - start_x))

            cutting_img_arr.append(img[start_y:end_y, start_x:end_x])

    if(count_y != 0):
        count_x = count_x_y//count_y
    else:
        print("error split")
        count_x = 0
    return (cutting_img_arr, ((count_x, count_y), size_data, overlay_coord_arr))

#old version
def overlay_glitting(pic_arr, data_splitting, size_step):
    (count_x, count_y), size_data, split_size = data_splitting
    size_pic = (size_data + split_size * (count_y-1), size_data + split_size * (count_x-1))
    out_pic = np.zeros(size_pic, np.float32)

    start_y_in_out = 0
    for y in range(count_y):
        if y != 0:
            offset_start_y = int((size_data-size_step)/2)
        else:
            offset_start_y = 0

        if y != count_y-1:
            offset_end_y = int((size_data-size_step) / 2)
        else:
            offset_end_y = 0

        end_y = size_data-offset_end_y
        end_y_in_out = start_y_in_out + end_y - offset_start_y

        start_x_in_out = 0
        for x in range(count_x):
            if x != 0:
                offset_start_x = int((size_data-size_step) / 2)
            else:
                offset_start_x = 0

            if x != count_x - 1:
                offset_end_x = int((size_data-size_step) / 2)
            else:
                offset_end_x = 0
            end_x = size_data-offset_end_x

            iter_img = pic_arr[y * count_x + x]

            end_x_in_out = start_x_in_out + end_x - offset_start_x

            #print("(",x,y,")")
            #print(start_y_in_out, end_y_in_out, ":" ,start_x_in_out, end_x_in_out)
            #print(offset_start_y, end_y, ":" ,offset_start_x, end_x)

            out_pic[start_y_in_out:end_y_in_out, start_x_in_out:end_x_in_out] = iter_img[offset_start_y:end_y, offset_start_x: end_x]

            start_x_in_out = end_x_in_out

        start_y_in_out = end_y_in_out

    return to_0_255_format_img(out_pic)


def glit_2_imgs(img1,img2, overlay, mode_x = True, mode_glit = "1/2"):
    if mode_x == True:
        h1,w1 = img1.shape[0:2]
        h2,w2 = img2.shape[0:2]
        if (h1 != h2):
            print("no glit img1 whis img2 from x")
            return None

        glit_img = np.zeros((h1, w1+w2-overlay), np.float32)

        #kernel glit
        if mode_glit == "1/2":
            glit_img[0:h1,0:w1-overlay//2] = img1[0:h1,0:w1-overlay//2]
            glit_img[0:h1, w1 - overlay // 2: w1+w2-overlay] = img2[0:h1, overlay // 2:w2]
        else:
            glit_img[0:h1, 0:w1] = img1[0:h1, 0:w1]
            glit_img[0:h1, w1: w1 + w2 - overlay] = img2[0:h1, overlay:w2]
            glit_img[0:h1, w1 - overlay: w1] += img2[0:h1, 0:overlay]

    else:
        h1, w1 = img1.shape[0:2]
        h2, w2 = img2.shape[0:2]
        if (w1 != w2):
            print("no glit img1 whis img2 from y")
            return None

        glit_img = np.zeros((h1 + h2 - overlay, w1), np.float32)

        # kernel glit
        if mode_glit == "1/2":
            glit_img[0:h1- overlay // 2, 0:w1] = img1[0:h1 - overlay // 2, 0:w1]
            glit_img[h1 - overlay // 2: h1 + h2 - overlay, 0:w1] = img2[overlay // 2:h2, 0:w1]
        else:
            glit_img[0:h1, 0:w1] = img1[0:h1, 0:w1]
            glit_img[h1: h1 + h2 - overlay,0:w1] = img2[overlay:h2, 0:w1]
            glit_img[h1 - overlay: h1, 0:w1] += img2[0:overlay, 0:w1]

    return glit_img

def overlay_glitting_with_superimposition(pic_arr, data_splitting):
    (count_x, count_y), size_data, overlay_coord_arr = data_splitting

    #sum_x_overlay = 0
    #for index in range(0, count_x - 1):
    #    #print(overlay_coord_arr[index][1])
    #    sum_x_overlay += overlay_coord_arr[index][1]
    #w = count_x * size_data - sum_x_overlay

    #sum_y_overlay = 0
    #for index in range(count_x-1, (count_x-1)*(count_y-1) + count_y - 1, count_x):
    #    sum_y_overlay += overlay_coord_arr[index][0]
    #    #print(overlay_coord_arr[index][0])
    #h = count_y * size_data - sum_y_overlay

    #size_pic = (h, w)
    #print(size_pic)

    glit_pic_line = []

    for i_y in range(0, count_y):
        glit_pic_line.append(pic_arr[i_y*count_x])
        for i in range(0, count_x - 1):
            glit_pic_line[i_y] = glit_2_imgs(glit_pic_line[i_y],
                                            pic_arr[i_y*count_x+i+1],
                                            overlay_coord_arr[i][1]) #, mode_x = True, mode_glit = "another")

    out_pic = glit_pic_line[0]
    for i,index_y in enumerate(range(count_x-1, (count_x-1)*(count_y-1) + count_y - 1, count_x)):
        #print(overlay_coord_arr[index_y][0])
        out_pic = glit_2_imgs(out_pic,
                              glit_pic_line[i+1],
                              overlay_coord_arr[index_y][0],
                              mode_x=False ) #,mode_glit="another summ overlay mode")

    return to_0_255_format_img(out_pic)


#old version
def overlay_mask_glitting(result, num_class, data_splitting, size_glit):
    union_result = []
    for i_class in range(num_class):
        pic = result[:,:,:,i_class]
        result_class = overlay_glitting(pic, data_splitting, size_glit)
        #print(result_class.shape)
        union_result.append(result_class)
    shape_img = union_result[0].shape
    #print(shape_img)
    union_arr = np.zeros(shape_img + (5,), np.uint8)
    for i_class in range(num_class):
        union_arr[:,:,i_class] = union_result[i_class]
    #print(union_arr.shape)

    return np.reshape(union_arr, (1,) + union_arr.shape)

def overlay_mask_glitting_with_superimposition(result, num_class, data_splitting):
    union_result = []
    for i_class in range(num_class):
        pic = result[:,:,:,i_class]
        result_class = overlay_glitting_with_superimposition(pic, data_splitting)
        #print(result_class.shape)
        union_result.append(result_class)
    shape_img = union_result[0].shape
    #print(shape_img)
    union_arr = np.zeros(shape_img + (num_class,), np.uint8)
    for i_class in range(num_class):
        union_arr[:,:,i_class] = union_result[i_class]
    #print(union_arr.shape)

    return np.reshape(union_arr, (1,) + union_arr.shape)

def preparation_for_CNN(cutting_img_arr):
    for img in cutting_img_arr:
        #img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img

def test_split_img_with_superimposition():
    img_name = "training0003.png"
    img = io.imread(os.path.join("data/test", img_name), as_gray=True)

    img = to_0_1_format_img(img)

    cutting_img_arr, data_splitting = overlay_splitting_with_superimposition(img, 256, 192, 64)

    (count_x, count_y), size_data, overlay_coord_arr = data_splitting

    print("c_x", count_x, "  c_y", count_y, " size_data ", size_data)

    for coord in overlay_coord_arr:
        print(coord)

    if not os.path.isdir("split_test"):
        print("создаю out_dir:" + "split_test")
        os.makedirs("split_test")

    io.imsave("split_test/" + img_name + ".png", img)
    for i, imgs in enumerate(cutting_img_arr):
        io.imsave("split_test/" + img_name + "_" + str(i) + ".png", imgs)

    #test = (to_0_255_format_img(img)- res_img)
    #cv2.imwrite("glit_test/" +"minus "+ img_name, test)
    #print(test.sum())

#old version
def test_glit_img():
    img_name = "training0003.png"
    img = io.imread(os.path.join("data/test", img_name), as_gray=True)

    img = to_0_1_format_img(img)

    cutting_img_arr, data_splitting = overlay_splitting(img, 256, 128)
    if not os.path.isdir("split_test"):
      print("создаю out_dir:" + "split_test")
      os.makedirs("split_test")

    io.imsave("split_test/" + img_name+".png", img)
    for i,imgs in enumerate(cutting_img_arr):
        io.imsave("split_test/"+ img_name + "_" +str(i) + ".png", imgs)

    test_list = []
    for i in range(117):
        if i%2 == 0:
            test_list.append(np.zeros((256,256), np.float32))
        else:
            test_list.append(np.ones((256,256), np.float32))

    res_img = overlay_glitting(test_list, data_splitting, 192)

    if not os.path.isdir("glit_test"):
       print("создаю out_dir:" + "glit_test")
       os.makedirs("glit_test")

    #test = (to_0_255_format_img(img)- res_img)
    #cv2.imwrite("glit_test/" +"minus "+ img_name, test)
    #print(test.sum())

    cv2.imwrite("glit_test/"+ img_name, res_img)

def test_glit_img_with_superimposition():
    img_name = "training0003.png"
    img = io.imread(os.path.join("data/test", img_name), as_gray=True)
    img = to_0_1_format_img(img)
    cutting_img_arr, data_splitting = overlay_splitting_with_superimposition(img, 256, 192, 64) #glit_size - 256-192 = 64

    (count_x, count_y), size_data, overlay_coord_arr = data_splitting
    print("c_x", count_x, "  c_y", count_y, " size_data ", size_data)
    #for coord in overlay_coord_arr:
    #    print(coord)

    if not os.path.isdir("split_test"):
        print("создаю out_dir:" + "split_test")
        os.makedirs("split_test")

    io.imsave("split_test/" + img_name + ".png", img)
    for i, imgs in enumerate(cutting_img_arr):
        io.imsave("split_test/" + img_name + "_" + str(i) + ".png", imgs)

    test_list = []
    for i in range(len(cutting_img_arr)):
        #test_list.append(np.full((256,256), 0.25, np.float32))

        if i%2 == 0:
            test_list.append(np.zeros((256,256), np.float32))
        else:
            test_list.append(np.ones((256,256), np.float32))


    res_img = overlay_glitting_with_superimposition(test_list, data_splitting)

    #if not os.path.isdir("glit_test"):
    #   print("создаю out_dir:" + "glit_test")
    #   os.makedirs("glit_test")
    #cv2.imwrite("glit_test/"+ img_name, res_img)




#main tailing function
def universal_tailing_tester_one_pic(nameCNN, num_class,img_name, size_data, size_step, minimum_overlay_size, save_dir, save_split_dir = None):
    model = unet(nameCNN, num_class = num_class)

    img = io.imread(os.path.join("data/test", img_name), as_gray=True)

    img = to_0_1_format_img(img)

    #io.imsave("split_test/" + img_name+".png", img)

    cutting_img_arr, data_splitting = overlay_splitting_with_superimposition(img, size_data, size_step, minimum_overlay_size)

    #SAVE SPLITE PIC
    if (save_split_dir is not None):
        if not os.path.isdir(save_split_dir):
           print("создаю out_dir_split:" + save_split_dir)
           os.makedirs(save_split_dir)


        for i,imgs in enumerate(cutting_img_arr):
            io.imsave(os.path.join(save_split_dir, img_name + "_" +str(i) + ".png"), imgs)

        with open(os.path.join(save_split_dir, img_name + "_split_logs.json"), 'w') as file:
            json.dump(data_splitting, file)

    cutting_generator = preparation_for_CNN(cutting_img_arr)

    results = model.predict(cutting_generator, batch_size=1, verbose=1)
    #print("results", results.shape)

    res_img = overlay_mask_glitting_with_superimposition(results, num_class, data_splitting)
    #print("overlay_mask_glitting", res_img.shape)

    saveResultMask(save_dir, res_img, [img_name], num_class=num_class)


def complex_tailing_test():
    list_CNN_name = ["my_unet_multidata_pe38_bs7_6class.hdf5",
                     "my_unet_multidata_pe38_bs7_5class.hdf5",
                     "my_unet_multidata_pe38_bs7_1class.hdf5"]
    list_CNN_num_class = [6,5,1]
    result_CNN_dir = ["data/result/CNN_6_class",
                      "data/result/CNN_5_class",
                      "data/result/CNN_1_class"]
    for i in range(len(list_CNN_num_class)):
        universal_tailing_tester_one_pic(nameCNN=list_CNN_name[i],
                                         num_class=list_CNN_num_class[i],
                                         img_name = "training0003.png",
                                         size_data=256,
                                         size_step=192,  # remove size_step / 2 pixels from each side
                                         minimum_overlay_size=64,  # max overlay size = size_data - minimum_overlay_size
                                         save_dir=result_CNN_dir[i])  # size_step should divide the size of the image(w,h)

'''
universal_tailing_tester_one_pic(nameCNN = "my_unet_multidata_pe38_bs7_5class.hdf5",
                    num_class = 5,
                    img_name="training0003.png",
                    size_data = 256,
                    size_step = 192,                #remove size_step / 2 pixels from each side
                    minimum_overlay_size = 64,      #max overlay size = size_data - minimum_overlay_size
                    save_dir= "data/test_result/",#size_step should divide the size of the image(w,h)
                    save_split_dir = "spliting")
'''


complex_tailing_test()


#test_split_img_with_superimposition()
#test_glit_img_with_superimposition()

#test_modul_v1()
#test_modul_v5()
#test_modul_v6()


#test_modul_v1_one_pic()
#test_modul_v5_one_pic()
#test_modul_v6_one_pic()