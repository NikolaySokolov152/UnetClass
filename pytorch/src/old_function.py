
'''
Файлопомойка для устаревших функций, которые не используются и которые не актуально поддерживать.
Но удалять данный код не хочется
'''



class Generator(Dataset):
    def __init__(self, dir_data, list_class_name, num_classes, transform_data, aug_dict, mode, save_inform, tailing,
                 augment):
        self.dir_data = dir_data
        self.list_class_name = list_class_name
        self.num_classes = num_classes
        self.transform_data = transform_data
        self.mode = mode
        self.save_inform = save_inform[0]
        self.inform_generator = save_inform[1]
        self.tailing = tailing
        self.augment = augment
        self.batch_size = self.transform_data.batch_size

        self.small_list_img_name = None

        if augment:
            if self.inform_generator == "validation generator":
                print("\nINFO: I'm a validation generator turn off augmentation for myself !\n")
                self.augment = False

        self.composition = create_transform(aug_dict, self.transform_data, self.augment)

        self.first_loop = True

    def __len__(self):
        'Denotes the number of batches per epoch'

        len_gen = int(np.floor(len(self.small_list_img_name) / self.transform_data.batch_size))
        if len_gen == 0:
            print("very small dataset, please add more img")

        return len_gen

    def __getitem__(self, index):
        # if self.inform_generator == "validation generator":
        #    print(index)
        if index >= self.__len__():
            raise StopIteration

        'Generate one batch of data'
        # read_names = self.small_list_img_name[index * self.transform_data.batch_size:(index+1) * self.transform_data.batch_size]
        #                                    len(self.small_list_img_name)]
        read_names = []
        for i in range(self.transform_data.batch_size):
            name = self.small_list_img_name[(index * self.transform_data.batch_size + i) % \
                                            len(self.small_list_img_name)]
            read_names.append(name)

        X, X_shapes = self.generate_X(read_names)
        if self.mode == 'train':
            y = self.generate_Y(read_names, X_shapes)
            X, y = self.augment_batch(X, y)

            X = torch.from_numpy(np.array(X)).permute(0, 3, 1, 2)
            y = torch.from_numpy(np.array(y)).permute(0, 3, 1, 2)
            return X, y

        elif self.mode == 'predict':
            return torch.from_numpy(np.array(X)).permute(0, 3, 1, 2)

        else:
            raise AttributeError('The mode parameter should be set to "train" or "predict".')

    def generate_X(self, list_name_in_batch):
        'Generates data containing batch_size samples'
        # Initialization
        X = []
        list_shape_X = []

        # Generate data
        for i, name in enumerate(list_name_in_batch):
            img_path = os.path.join(self.dir_data.dir_img_name, name)

            if self.transform_data.color_mode_img == "rgb":
                img = io.imread(img_path)
            elif self.transform_data.color_mode_img == "hsv":
                image = io.imread(img_path)
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
            else:
                image = io.imread(img_path, as_gray=True)
                img = np.expand_dims(image, axis=-1)

            # print(img.shape)
            img = img.astype(np.float32)
            img = to_0_1_format_img(img)
            # Store samples
            X.append(img)
            list_shape_X.append(img.shape[:2])

        return X, list_shape_X

    def generate_Y(self, list_name_in_batch, X_shapes):
        y = []

        if self.transform_data.mode_mask == 'separated':
            for j, name in enumerate(list_name_in_batch):

                y_arr = np.empty((*X_shapes[j], self.num_classes), np.float32)

                for i in range(self.num_classes):
                    img_path = os.path.join(self.dir_data.dir_mask_name, self.list_class_name[i],
                                            self.dir_data.add_mask_prefix + name)

                    if os.path.isfile(img_path):
                        image = io.imread(img_path, as_gray=True)
                        masks = image.astype(np.float32)
                        masks = to_0_1_format_img(masks)
                        if self.transform_data.binary_mask:
                            masks[masks[:,:]<0.5]=0
                            masks[masks[:,:]>0.4]=1
                    else:
                        if self.first_loop:
                            print("no open ", img_path)
                        masks = np.zeros(X_shapes[j], np.float32)
                    # print(masks.shape)

                    y_arr[:, :, i] = masks

                y.append(y_arr)

        elif self.transform_data.mode_mask == 'image':
            for j, name in enumerate(list_name_in_batch):
                img_path = os.path.join(self.dir_data.dir_mask_name, self.list_class_name[0],
                                        self.dir_data.add_mask_prefix + name)
                if os.path.isfile(img_path):
                    if self.transform_data.color_mode_img == "rgb":
                        img = io.imread(img_path)
                    elif self.transform_data.color_mode_img == "hsv":
                        image = io.imread(img_path)
                        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
                    else:
                        image = io.imread(img_path, as_gray=True)
                        img = np.expand_dims(image, axis=-1)

                    masks = img.astype(np.float32)
                    masks = to_0_1_format_img(masks)
                else:
                    print("no open ", img_path)
                    raise Exception(f"ERROR! No open: {img_path}")
                # print(masks.shape)
            y.append(masks)
        elif self.transform_data.mode_mask == 'no_mask':
            # заглушка
            for j, name in enumerate(list_name_in_batch):
                y.append(np.zeros(X_shapes[j], np.float32))
        else:
            raise AttributeError('The "mode_mask" parameter should be set to "separated", "image" or "no_mask" otherwise not implemented.')

        return y

    def random_transform_one_frame(self, img, masks):
        composed = self.composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

    def augment_batch(self, img_batch, masks_batch):
        for i in range(self.transform_data.batch_size):
            img_batch[i], masks_batch[i] = self.random_transform_one_frame(img_batch[i], masks_batch[i])

        return img_batch, masks_batch

    def on_epoch_end(self):
        self.first_loop = False
    #    print()
    ##    print(self.inform_generator)
    ##    print(self.small_list_img_name)
    #    print()

class DataGenerator():
    'Load and generate data'

    def __init__(self, dir_data=InfoDirData(),
                 list_class_name=["1", "2", "3", "4"], num_classes=2, mode='train', tailing=False,
                 aug_dict=dict(), augment=False, shuffle=True, seed=42, subsampling="crossover",
                 transform_data=TransformData(), save_inform=SaveData(), share_validat=0.2, silence_mode=False):

        self.typeGen = "DataGenerator"

        self.dir_data = dir_data
        self.list_class_name = list_class_name
        self.num_classes = num_classes
        self.transform_data = transform_data
        self.aug_dict = aug_dict
        self.mode = mode
        self.subsampling = subsampling

        self.save_inform = save_inform

        self.share_val = share_validat
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed

        self.tailing = tailing

        self.silence_mode = silence_mode

        img_type = ('.png', '.jpg', '.jpeg')
        self.list_img_name = [name for name in os.listdir(dir_data.dir_img_name) if name.endswith((img_type))]
        print("In normal dir:", len(self.list_img_name), "images", flush=True)

        random.shuffle(self.list_img_name)

        if self.mode == "train":
            self.gen_train = Generator(dir_data, list_class_name, num_classes, transform_data, aug_dict, mode,
                                       [save_inform, "train generator"], tailing, augment)
            self.gen_valid = Generator(dir_data, list_class_name, num_classes, transform_data, aug_dict, mode,
                                       [save_inform, "validation generator"], tailing, augment)

        else:
            self.gen_test = Generator(dir_data, list_class_name, num_classes, transform_data, aug_dict, mode,
                                      [save_inform, "test generator"], tailing, False)

        # self.indexes = [i for i in range(len(self.list_img_name))]

        np.random.seed(self.seed)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        # print(int(len(self.list_img_name) * self.share_val))
        # print(int(len(self.list_train_img_name) / self.transform_data.batch_size))
        # print(int(len(self.list_validation_img_name) / self.transform_data.batch_size))

        len_gen = int(np.floor(len(self.list_img_name) / self.transform_data.batch_size))
        if len_gen == 0:
            print("very small dataset, please add more img")

        return len_gen

    def load_new_data(self, img_names):
        imgs = []
        masks_glob = []

        for name in img_names:
            img_path = os.path.join(self.dir_data.dir_img_name, name)

            if self.transform_data.color_mode_img == "rgb":
                image = io.imread(img_path)
                img = cv2.resize(image, self.transform_data.target_size)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                image = io.imread(img_path, as_gray=True)
                image = np.expand_dims(image, axis=-1)
                img = cv2.resize(image, self.transform_data.target_size)

            # print(img.shape)
            img = img.astype(np.float32)
            img = to_0_1_format_img(img)

            # Store samples
            imgs.append(img)

        if self.transform_data.mode_mask == 'separated':
            for name in img_names:
                class_mask = []
                for i in range(self.num_classes):
                    img_path = os.path.join(self.dir_data.dir_mask_name, self.list_class_name[i],
                                            self.dir_data.add_mask_prefix + name)

                    if os.path.isfile(img_path):
                        image = io.imread(img_path, as_gray=True)
                        masks = image.astype(np.float32)
                        masks = to_0_1_format_img(masks)
                    else:
                        print("no open ", img_path)
                        masks = np.zeros(self.transform_data.target_size)

                    class_mask.append(masks)
                masks_glob.append(class_mask)

        elif self.transform_data.mode_mask == 'image':
            for name in img_names:
                img_path = os.path.join(self.dir_data.dir_mask_name,
                                        self.dir_data.add_mask_prefix + name)

                if os.path.isfile(img_path):
                    if self.transform_data.color_mode_img == "rgb":
                        img = io.imread(img_path)
                    elif self.transform_data.color_mode_img == "hsv":
                        image = io.imread(img_path)
                        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
                    else:
                        image = io.imread(img_path, as_gray=True)
                        img = np.expand_dims(image, axis=-1)

                    masks = img.astype(np.float32)
                    masks = to_0_1_format_img(masks)
                else:
                    print("no open ", img_path)
                    raise Exception(f"ERROR! No open: {img_path}")
                # print(masks.shape)
                masks_glob.append(masks)
        elif self.transform_data.mode_mask == 'no_mask':
            # заглушка
            for i in range(len(img_names)):
                masks_glob.append(np.zeros(self.transform_data.target_size, np.float32))
        else:
            raise AttributeError('The "mode_mask" parameter should be set to "separated", "image" or "no_mask" otherwise not implemented.')

        return imgs, masks_glob

    def on_epoch_end(self):
        size_val = int(len(self.list_img_name) * self.share_val)
        'Updates indexes after each epoch'
        # print("change generator")

        if self.mode == 'train':
            self.gen_train.small_list_img_name = self.list_img_name[:len(self.list_img_name) - size_val]
            self.gen_valid.small_list_img_name = self.list_img_name[len(self.list_img_name) - size_val:]

            # print(self.indexes[:len(self.list_img_name) - size_val])
            # print(self.indexes[len(self.list_img_name) - size_val:])


        elif self.mode == 'predict':
            self.gen_test.small_list_img_name = self.list_img_name
        else:
            raise AttributeError('The "mode" parameter should be set to "train" or "predict".')

        if self.shuffle is True:
            print("shuffling in the generator")
            # print()

            if self.subsampling == 'random':
                # list_shuffle_indexes_train = self.indexes[:len(self.list_img_name) - size_val]
                # list_shuffle_indexes_test  = self.indexes[len(self.list_img_name) - size_val:]

                # np.random.shuffle(list_shuffle_indexes_train)
                # np.random.shuffle(list_shuffle_indexes_test)

                # self.indexes = list_shuffle_indexes_test + list_shuffle_indexes_train

                list_shuffle_train = self.list_img_name[:len(self.list_img_name) - size_val]
                list_shuffle_test = self.list_img_name[len(self.list_img_name) - size_val:]

                np.random.shuffle(list_shuffle_train)
                np.random.shuffle(list_shuffle_test)

                self.list_img_name = list_shuffle_test + list_shuffle_train


            elif self.subsampling == 'crossover':
                # self.indexes = self.indexes[len(self.list_img_name) - size_val:] + self.indexes[:len(self.list_img_name) - size_val]
                self.list_img_name = self.list_img_name[len(self.list_img_name) - size_val:] + self.list_img_name[:len(
                    self.list_img_name) - size_val]

            else:
                raise AttributeError('The "subsampling" parameter should be set to "random" or "crossover".')
