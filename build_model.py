import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

class Resblock(Layer):
    def __init__(self, chn, strides):
        super(Resblock, self).__init__()
        self.chn = chn
        self.strides = strides

    def build(self, input_shape):
        self.conv1 = Conv2D(self.chn, 3, strides=self.strides, padding='same', activation=LeakyReLU(0.3))
        self.bn = BatchNormalization()
        self.conv2 = Conv2D(self.chn, 3, padding='same', activation=LeakyReLU(0.3))
        self.conv3 = Conv2D(self.chn, 3, strides=self.strides, padding='same', activation=None)

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        if inputs.shape[-1] == self.chn:
            return tf.math.add(inputs, conv2)
        else:
            conv3 = self.conv3(inputs)
            return tf.math.add(conv2, conv3)

def build_relight():
    inputs = Input((128,128,2))
    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(inputs)  # 128 -> 64
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn1) # 64 -> 32
    bn2 = BatchNormalization()(conv2)
    resblock1 = Resblock(chn=256, strides=(2, 2))(bn2) # 32 -> 16
    bn3 = BatchNormalization()(resblock1)
    resblock2 = Resblock(chn=256, strides=(1, 1))(bn3)
    bn4 = BatchNormalization()(resblock2)
    resblock3 = Resblock(chn=256, strides=(1, 1))(bn4)
    bn5 = BatchNormalization()(resblock3)
    resblock4 = Resblock(chn=256, strides=(1, 1))(bn5)
    bn6 = BatchNormalization()(resblock4)
    resblock5 = Resblock(chn=256, strides=(1, 1))(bn6)
    bn7 = BatchNormalization()(resblock5)
    resblock6 = Resblock(chn=256, strides=(1, 1))(bn7)
    bn8 = BatchNormalization()(resblock6)
    dconv1 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn8)
    bn9 = BatchNormalization()(dconv1)
    dconv2 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn9)
    bn10 = BatchNormalization()(dconv2)
    dconv3 = Conv2DTranspose(1, 3, strides=(2, 2), padding='same', activation='tanh')(bn10)
    model = Model(inputs, dconv3)
    model.summary()
    return model

def build_discriminator():
    inputs = Input((128,128,1))
    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(inputs)  # 128 -> 64
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(conv1)  # 64 -> 32
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(conv2)  # 32 -> 16
    conv4 = Conv2D(512, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(conv3)  # 16 -> 8
    flat = Flatten()(conv4)
    classified = Dense(46, activation='softmax')(flat)
    validation = Conv2D(1, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(conv4)

    model = Model(inputs, [validation, classified])
    model.summary()
    return model

class Conv_module(Layer):
    def __init__(self, chn):
        super(Conv_module, self).__init__()
        self.conv1 = Conv2D(chn, 4, padding='same', strides=(2,2))
        self.leaky = LeakyReLU(0.01)
        self.conv2 = Conv2D(chn, 3, padding='same', strides=(1,1))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky(x)
        return self.conv2(x)

def build_vsn(input_shape = (128,128,1)):
    inputs = Input(input_shape)
    conv1 = Conv_module(64)(inputs)
    conv2 = Conv_module(128)(conv1)
    conv3 = Conv_module(256)(conv2)
    conv4 = Conv_module(256)(conv3)
    conv5 = Conv_module(256)(conv4)
    conv6 = Conv_module(256)(conv5)
    conv7 = Conv2D(2, 2, activation='softmax')(conv6)
    conv7 = Flatten()(conv7)

    model = Model(inputs, [conv2, conv7])
    model.summary()
    return model

if __name__ == '__main__':
    vsn = build_vsn()
    vsn.summary()
    print(vsn.output)
