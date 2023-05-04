from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.applications.mobilenet_v2 import MobileNetV2


def unet(input_size = (256,256,1), num_class = 2):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    normal1 = (BatchNormalization())(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal1)
    normal1 = (BatchNormalization())(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normal1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    normal2 = (BatchNormalization())(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal2)
    normal2 = (BatchNormalization())(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normal2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    normal3 = (BatchNormalization())(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal3)
    normal3 = (BatchNormalization())(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normal3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    normal4 = (BatchNormalization())(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal4)
    normal4 = (BatchNormalization())(conv4)
    drop4 = Dropout(0.5)(normal4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    normal5 = (BatchNormalization())(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal5)
    normal5 = (BatchNormalization())(conv5)
    drop5 = Dropout(0.5)(normal5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    normal6 = (BatchNormalization())(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal6)
    normal6 = (BatchNormalization())(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    normal7 = (BatchNormalization())(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal7)
    normal7 = (BatchNormalization())(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    normal8 = (BatchNormalization())(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal8)
    normal8 = (BatchNormalization())(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    normal9 = (BatchNormalization())(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal9)
    normal9 = (BatchNormalization())(conv9)

    #dense10 = Dense(num_class)(normal9)
    #conv10 = Activation('sigmoid')(dense10)
    
    out = Conv2D(num_class, (1, 1), padding="same")(normal9)
    conv10 = Activation("sigmoid")(out)

    model = Model(inputs=inputs, outputs=conv10)

    return model
    
def tiny_unet(input_size=(256, 256, 1), num_class=2):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    normal1 = (BatchNormalization())(conv1)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal1)
    normal1 = (BatchNormalization())(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normal1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    normal2 = (BatchNormalization())(conv2)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal2)
    normal2 = (BatchNormalization())(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normal2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    normal3 = (BatchNormalization())(conv3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal3)
    normal3 = (BatchNormalization())(conv3)
    drop3 = Dropout(0.5)(normal3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    normal4 = (BatchNormalization())(conv4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal4)
    normal4 = (BatchNormalization())(conv4)
    drop4 = Dropout(0.5)(normal4)

    up5 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([drop3, up5], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    normal5 = (BatchNormalization())(conv5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal5)
    normal5 = (BatchNormalization())(conv5)

    up6 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    normal6 = (BatchNormalization())(conv6)
    conv6 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal6)
    normal6 = (BatchNormalization())(conv6)

    up7 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    normal7 = (BatchNormalization())(conv7)
    conv7 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal7)
    normal7 = (BatchNormalization())(conv7)

    #dense8 = Dense(num_class)(normal7)
    #conv8 = Activation('sigmoid')(dense8)
    
    conv8 = Conv2D(num_class, (1, 1), padding="same")(normal7)
    conv8 = Activation('sigmoid')(conv8)

    model = Model(inputs=inputs, outputs=conv8)

    return model

def tiny_unet_v3(input_size=(256, 256, 1), num_class=2):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    normal1 = (BatchNormalization())(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal1)
    normal1 = (BatchNormalization())(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normal1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    normal2 = (BatchNormalization())(conv2)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal2)
    normal2 = (BatchNormalization())(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normal2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    normal3 = (BatchNormalization())(conv3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal3)
    normal3 = (BatchNormalization())(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normal3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    normal4 = (BatchNormalization())(conv4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal4)
    normal4 = (BatchNormalization())(conv4)
    drop4 = Dropout(0.5)(normal4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    normal5 = (BatchNormalization())(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal5)
    normal5 = (BatchNormalization())(conv5)
    drop5 = Dropout(0.5)(normal5)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    normal6 = (BatchNormalization())(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal6)
    normal6 = (BatchNormalization())(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    normal7 = (BatchNormalization())(conv7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal7)
    normal7 = (BatchNormalization())(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    normal8 = (BatchNormalization())(conv8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal8)
    normal8 = (BatchNormalization())(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    normal9 = (BatchNormalization())(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal9)
    normal9 = (BatchNormalization())(conv9)
    
    #dense10 = Dense(num_class, activation='sigmoid')(normal9)
    
    out = Conv2D(num_class, (1, 1), padding="same")(normal9)
    dense10 = Activation("sigmoid")(out)

    model = Model(inputs=inputs, outputs=dense10)
    
    return model


def mobile_unet_v2(input_size=(256, 256, 1), num_class=2):

    inputs = Input(shape=input_size, name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights=None, include_top=False, alpha=1)
    
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    out = Conv2D(num_class, (1, 1), padding="same")(x)
    out = Activation("sigmoid")(out)
    
    model = Model(inputs, outputs = out)

    return model
