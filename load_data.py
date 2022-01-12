import os
import cv2
import numpy as np
from random import *

def load_yaleb_train(path):
    light_type = os.listdir(path)
    light_type.sort()
    dataset = []
    for light in light_type:
        subpath = path + '/' + light
        img_name = os.listdir(subpath)
        img_name.sort()
        for name in img_name:
            img = cv2.imread(subpath + '/' + name)
            img = cv2.resize(img, (128, 128))
            dataset.append(img)
    dataset = np.array(dataset).astype('float32')
    dataset = dataset/127.5 - 1
    return dataset

def load_img_cond1(target_light_type, train=True):

    path = '/home/pomelo96/Desktop/datasets/Yaleb/'
    if train : path += 'train'
    else     : path += 'test'
    GT = load_light_type(target_light_type, train=train)
    reference = GT

    light_name = os.listdir(path)
    light_name.sort()
    dataset = []
    for i in range(len(light_name)):
        if i != target_light_type:
            light = light_name[i]
            subpath = path + '/' + light
            img_name = os.listdir(subpath)
            img_name.sort()
            for name in img_name:
                img = cv2.imread(subpath + '/' + name)
                img = cv2.resize(img, (128, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=-1)
                dataset.append(img)
    dataset = np.array(dataset).astype('float32')
    dataset = dataset / 127.5 - 1
    return dataset, reference, GT

def load_light_type(light_type, train=True):
    dataset = []
    path = '/home/pomelo96/Desktop/datasets/Yaleb/'
    if train: path += 'train'
    else    : path += 'test'
    light_name = os.listdir(path)
    light_name.sort()
    light_name = light_name[light_type]
    subpath = path+ '/'+ light_name
    img_name = os.listdir(subpath)
    img_name.sort()
    for name in img_name:
        img = cv2.imread(subpath + '/' +name)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        dataset.append(img)
    dataset = np.array(dataset).astype('float32')
    dataset = dataset / 127.5 - 1
    return dataset

def load_id(id, train=True):
    dataset = []
    path = '/home/pomelo96/Desktop/datasets/Yaleb/'
    if train : path += 'train'
    else     : path += 'test'
    light_type = os.listdir(path)
    light_type.sort()
    dataset = []
    for light in light_type:
        subpath = path + '/' + light
        img_name = os.listdir(subpath)
        img_name.sort()
        name = img_name[id]
        img = cv2.imread(subpath + '/' + name)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        dataset.append(img)
    dataset = np.array(dataset).astype('float32')
    dataset = dataset / 127.5 - 1
    return dataset

def load_ck():
    dataset = []
    path = 'testing_ck_img'
    testing_img_name = os.listdir(path)
    for name in testing_img_name:
        img = cv2.imread(path + '/' + name)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        dataset.append(img)
    dataset = np.array(dataset).astype('float32')
    dataset = dataset / 127.5 - 1

    return dataset

def load_referennce_cond7(emo):
    dataset = []
    if emo == 'natural':
        path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK/train/Natural image'



