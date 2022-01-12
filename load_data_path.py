import glob
import cv2
import numpy as np
import random
import os

def load_image(roots):
    dataset = []
    for root in roots:
        img = cv2.imread(root)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img,axis=-1)
        dataset.append(img)
    return (np.array(dataset).astype('float32'))/127.5 - 1

#load all roots(path) only, not prepare datasets
def load_YaleB():
    train_roots,test_roots = [], []
    train_id, test_id = [],[]
    train_light, test_light = [], []

    file_root = '/home/pomelo96/Desktop/datasets/Yaleb/train/'
    light_label = 0
    for light in sorted(glob.glob(file_root + '*')):
        id = 0
        for image_root in sorted(glob.glob(light + '/*.jpg')):
            train_roots.append(image_root)
            train_id.append(id)
            train_light.append(light_label)
            id += 1
        light_label += 1

    file_root = '/home/pomelo96/Desktop/datasets/Yaleb/test/'
    light_label = 0
    for light in sorted(glob.glob(file_root + '*')):
        id = 32
        for image_root in sorted(glob.glob(light + '/*.jpg')):
            test_roots.append(image_root)
            test_id.append(id)
            test_light.append(light_label)
            id += 1
        light_label += 1
    return train_roots, train_id, train_light, test_roots, test_id, test_light

# train_roots, train_id, train_light, test_roots, test_id, test_light = load_YaleB()

def get_batch_data(data,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1) * batch_size
    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min, range_max))
    temp_data = [data[idx] for idx in index]
    return temp_data

def set_data_for_cycleGAN(roots, light_list, is_pretrain):
    # print(len(roots),len(ID_list),len(light_list))
    light_roots = [[], [], [], [], [], [], [], [], [], [], []]
    for idx in range(len(roots)):
        root = roots[idx]
        light_label = light_list[idx]
        light_roots[light_label].append(root)
    #把圖片的路徑依光線的類別放在各自的list
    #例:圖片1為光線2 圖片1的roots放在light_roots的第二個list
    #全部的圖片路徑都會被丟進去

    input_image_roots,reference_image_roots,GT_image_roots,id_class = [],[],[],[]
    if is_pretrain:
        for light in light_roots:
            id = 0
            for img_root in light:
                for ref_img_root in light:
                    if img_root != ref_img_root:
                        input_image_roots.append(img_root)
                        GT_image_roots.append(img_root)
                        reference_image_roots.append(ref_img_root)
                        id_class.append(id)
                id += 1

    else:
        for light_label_for_input in range(11):
            for light_label_for_reference in range(11):
                if light_label_for_input != light_label_for_reference:
                    input_light_roots = light_roots[light_label_for_input]
                    reference_light_roots = light_roots[light_label_for_reference]
                    i = 0
                    for input_root in input_light_roots:
                        reference_idx = random.sample(range(len(reference_light_roots)),6) #training is 10
                        idx_times = 0
                        for idx in reference_idx:
                            reference_root = reference_light_roots[idx]
                            input_ID = input_root.split('/')[-1].split('_')[0]
                            reference_ID = reference_root.split('/')[-1].split('_')[0]
                            if input_ID != reference_ID:
                                input_image_roots.append(input_root)
                                reference_image_roots.append(reference_root)
                                input_light = input_root.split('/')[-2]
                                output_light = reference_root.split('/')[-2]
                                GT_root = input_root.replace(input_light,output_light)
                                GT_image_roots.append(GT_root)
                                id_class.append(i)
                                idx_times+=1
                            if idx_times > 3:
                                break
                        i+=1

    temp = list(zip(input_image_roots, reference_image_roots, GT_image_roots, id_class))
    random.shuffle(temp)
    input_image_roots, reference_image_roots, GT_image_roots, id_class = zip(*temp)

    return input_image_roots,reference_image_roots,GT_image_roots,id_class

if __name__ == '__main__':
    train_roots, train_id, train_light, test_roots, test_id, test_light = load_YaleB()
    train_roots = np.array(train_roots)
    '''
    input_image_roots,reference_image_roots,GT_image_roots,id_class = set_data_for_cycleGAN(train_roots,train_light,is_pretrain=False)
    for idx in range(10):
        print('')
        print(input_image_roots[idx].split('/')[-1])
        print(reference_image_roots[idx].split('/')[-1])
        print(GT_image_roots[idx].split('/')[-1])
        print(id_class[idx])
    print(len(input_image_roots))
    '''