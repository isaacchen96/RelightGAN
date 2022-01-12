import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.applications.vgg16 import VGG16

class style_loss():
    def __init__(self):
        self.vgg16_pool1 = self.build_vgg_extract('block1_pool')
        self.vgg16_pool2 = self.build_vgg_extract('block2_pool')
        self.vgg16_pool3 = self.build_vgg_extract('block3_pool')
        self.vgg16_pool4 = self.build_vgg_extract('block4_pool')

    def build_vgg_extract(self, layer_name):
        vgg16 = VGG16(input_shape=(128, 128, 1), include_top=False, weights=None)
        model = Model(inputs=vgg16.input, outputs=vgg16.get_layer(layer_name).output)
        return model

    def gram_matrix(self, mat):
        return tf.linalg.matmul(a=mat, b=mat, transpose_a=True)

    def predict_loss(self, reference, gen_img):
        mae = MeanAbsoluteError()
        loss_pool1 = mae(self.gram_matrix(self.vgg16_pool1(reference)),
                            self.gram_matrix(self.vgg16_pool1(gen_img)))

        loss_pool2 = mae(self.gram_matrix(self.vgg16_pool2(reference)),
                         self.gram_matrix(self.vgg16_pool2(gen_img)))

        loss_pool3 = mae(self.gram_matrix(self.vgg16_pool3(reference)),
                         self.gram_matrix(self.vgg16_pool3(gen_img)))

        loss_pool4 = mae(self.gram_matrix(self.vgg16_pool4(reference)),
                         self.gram_matrix(self.vgg16_pool4(gen_img)))

        return tf.reduce_mean([loss_pool1, loss_pool2, loss_pool3, loss_pool4])

def adversarial_loss(target, pred):
    if target:
        valid = tf.ones_like(pred)
    else:
        valid = tf.zeros_like(pred)
    mse = MeanSquaredError()
    return mse(valid, pred)

def classify_loss(gt, pred):
    cce = CategoricalCrossentropy()
    return cce(gt, pred)

def img_loss(gt, pred):
    mae = MeanAbsoluteError()
    return mae(gt, pred)

def PSNR_loss(x, y):
    max_x = tf.reduce_max(x,axis = (1,2,3))
    max_y = tf.reduce_max(y,axis = (1,2,3))
    max_value = tf.math.maximum(max_x,max_y)

    metric = tf.reduce_sum((x - y)*(x - y), axis=(1, 2, 3))
    loss = 0.1 * tf.math.log((max_value * max_value)/metric)
    return -tf.reduce_mean(loss)

def SSIM_loss(x, y):
    '''
    mean_x = tf.reduce_mean(x, axis=(-1, -2, -3))
    mean_y = tf.reduce_mean(y, axis=(-1, -2, -3))
    var_x = tfp.stats.covariance(x, x, sample_axis=(-2, -3), event_axis=None)
    var_y = tfp.stats.covariance(y, y, sample_axis=(-2, -3), event_axis=None)
    cov_xy = tfp.stats.covariance(x, y, sample_axis=(-2, -3), event_axis=None)

    C1 = 0.0001
    C2 = 0.1
    var_x = tf.reduce_sum(var_x, axis=-1)
    var_y = tf.reduce_sum(var_y, axis=-1)
    cov_xy = tf.reduce_sum(cov_xy, axis=-1)
    up = tf.multiply((2 * mean_x * mean_y + C1), (2 * cov_xy + C2))
    down = tf.multiply((mean_x*mean_x + mean_y*mean_y + C1), (var_x*var_x + var_y*var_y + C2))

    return 1 - up/down
    '''
    x = 0.5 * (x + 1)
    y = 0.5 * (y + 1)
    return 1 - tf.image.ssim(x, y, max_val=1, filter_size=4, filter_sigma=1.5, k1=0.01, k2=0.03)

def image_to_batch(x, size=4):
    x =  tf.image.extract_patches(images=x,
                           sizes=[1, size, size, 1],
                           strides=[1, size, size, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
    x = tf.reshape(x, (x.shape[0], -1, size, size, 1))
    return x

def SSIM_att_loss(x, y, att):
    x = image_to_batch(x, size=4)
    y = image_to_batch(y, size=4)
    ssim = SSIM_loss(x, y)

    att = tf.reshape(att, (att.shape[0], att.shape[1], att.shape[2], 1))
    att = image_to_batch(att, size=1)
    att = tf.reshape(att, (att.shape[0], att.shape[1]))

    att_min = tf.reduce_min(att, axis=-1)
    att_max = tf.reduce_max(att, axis=-1)
    att_min = tf.reshape(att_min, (att_min.shape[0], 1))
    att_max = tf.reshape(att_max, (att_max.shape[0], 1))
    att = (att - att_min) / (att_max - att_min)

    loss = tf.reduce_sum(tf.multiply(ssim, att), axis=-1)
    norm = tf.reduce_sum(att, axis=-1)

    return tf.reduce_mean(tf.divide(loss, norm))
