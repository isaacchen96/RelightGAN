from build_model import *
from load_data import *
from train import *
import numpy as np
import matplotlib.pyplot as plt
from random import *
import cv2
from CK_work import *

if __name__ == '__main__':
    condition = 13
    relight_cycle_generator = build_relight()
    relight_cycle_generator.load_weights('weight_ck/ck_generator_part2_weights_13')

    if condition == 0:   #testing data random test

        path = '/home/pomelo96/Desktop/datasets/Yaleb/test'
        _, _, _, train_roots, train_id, train_light = load_YaleB()
        input_image_roots, reference_image_roots, GT_image_roots, id_class \
            = set_data_for_cycleGAN(train_roots, train_light, is_pretrain=False)
        for i in range(10):
            source_sampling = load_image(get_batch_data(input_image_roots, i, 10))
            reference_sampling = load_image(get_batch_data(reference_image_roots, i, 10))
            gt_sampling = load_image(get_batch_data(GT_image_roots, i, 10))
            label = get_batch_data(id_class, i, 10)

            gen_imgs_1 = relight_cycle_generator.predict(tf.concat([source_sampling, reference_sampling], axis=-1))
            gen_imgs_2 = relight_cycle_generator.predict(tf.concat([gen_imgs_1, source_sampling], axis=-1))
            gen_imgs_1 = 0.5 * (gen_imgs_1 + 1)
            gen_imgs_2 = 0.5 * (gen_imgs_2 + 1)
            reference_sampling = 0.5 * (reference_sampling+1)
            source_sampling = 0.5 * (source_sampling+1)
            gt_sampling = 0.5 * (gt_sampling+1)

            r, c = 5, 10
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0
            for k in range(c):
                axs[0, k].imshow(source_sampling[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_imgs_1[cnt], cmap='gray')
                axs[1, k].axis('off')
                axs[2, k].imshow(gt_sampling[cnt], cmap='gray')
                axs[2, k].axis('off')
                axs[3, k].imshow(reference_sampling[cnt], cmap='gray')
                axs[3, k].axis('off')
                axs[4, k].imshow(gen_imgs_2[cnt], cmap='gray')
                axs[4, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond0/test_{}.jpg'.format(i))
            plt.close()

#-------------------------------------------------------------------------------------#

    elif condition==1: #Input: ID any light any -> Output ID any light same

        for target_light in range(6, 11):
            source, reference, gt = load_img_cond1(target_light_type=target_light, train=False)
            for i in range(10):
                source_ = source[6 * i:6 * (i + 1)]
                start = randint(1, 5)
                reference_index = [(start + i) % 6 for i in range(6)]
                reference_ = reference[reference_index]
                gen_imgs = relight_cycle_generator.predict(tf.concat([source_, reference_], axis=-1))

                gen_imgs = 0.5 * (gen_imgs + 1)
                source_ = 0.5 * (source_ + 1)
                reference_ = 0.5 * (reference_ + 1)
                gt_ = 0.5 * (gt + 1)

                #for j in range(4):
                r, c = 4, 6
                fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
                plt.subplots_adjust(hspace=0.2)
                cnt = 0
                for k in range(c):
                    axs[0, k].imshow(source_[cnt], cmap='gray')
                    axs[0, k].axis('off')
                    axs[1, k].imshow(gen_imgs[cnt], cmap='gray')
                    axs[1, k].axis('off')
                    axs[2, k].imshow(gt_[cnt], cmap='gray')
                    axs[2, k].axis('off')
                    axs[3, k].imshow(reference_[cnt], cmap='gray')
                    axs[3, k].axis('off')

                    cnt += 1
                fig.savefig('picture/cond1/test_T{}_S{}.png'.format(target_light, i))
                plt.close()

    elif condition == 2:        #source light = reference light => predict must same with source

        for target_light_type in range(11):
            source = load_light_type(target_light_type, train=False)
            start = randint(1, 5)
            reference_index = [(start+i) % 6 for i in range(6)]
            reference = source[reference_index]

            gen_imgs = relight_cycle_generator.predict(tf.concat([source, reference], axis=-1))

            gen_imgs = 0.5 * (gen_imgs + 1)
            source_ = 0.5 * (source + 1)
            reference = 0.5 * (reference + 1)

            r, c = 3, 6
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0
            for k in range(c):
                axs[0, k].imshow(source[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_imgs[cnt], cmap='gray')
                axs[1, k].axis('off')
                axs[2, k].imshow(reference[cnt], cmap='gray')
                axs[2, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond2/test_T{}.png'.format(target_light_type))
            plt.close()

    elif condition == 3:    #same id with all lights -> same id same 1light
        for id_source in range(6):

            for target_light in range(11):
                source = load_id(id_source, train=False)
                reference = load_light_type(target_light, train=True)
                gt = reference[id_source]
                reference = np.delete(reference, [id_source], axis=0)
                start = randint(1, 5)
                reference_index = [(start + i) % 6 for i in range(11)]
                reference_ = reference[reference_index]
                gen_imgs = relight_cycle_generator.predict(tf.concat([source, reference_], axis=-1))

                gen_imgs = 0.5 * (gen_imgs + 1)
                source = 0.5 * (source + 1)
                reference_ = 0.5 * (reference_ + 1)

                r, c = 4, 6
                fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
                plt.subplots_adjust(hspace=0.2)
                cnt = 0
                for k in range(c):
                    axs[0, k].imshow(source[cnt], cmap='gray')
                    axs[0, k].axis('off')
                    axs[1, k].imshow(gen_imgs[cnt], cmap='gray')
                    axs[1, k].axis('off')
                    axs[2, k].imshow(gt, cmap='gray')
                    axs[2, k].axis('off')
                    axs[3, k].imshow(reference_[cnt] , cmap='gray')
                    axs[3, k].axis('off')

                    cnt += 1
                fig.savefig('picture/cond3/test_ID{}_light{}.png'.format(id_source, target_light))
                plt.close()

    elif condition == 4:   #reference all black or all white
        for id_source in range(6):

            source = load_id(id_source, train=False)
            reference = np.ones_like(source)
            inputs = tf.concat([source, reference], axis=-1)
            gen_imgs = relight_cycle_generator.predict(tf.concat([source, reference], axis=-1))

            gen_imgs = 0.5 * (gen_imgs + 1)
            source = 0.5 * (source + 1)

            r, c = 2, 11
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0
            for k in range(c):
                axs[0, k].imshow(source[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_imgs[cnt], cmap='gray')
                axs[1, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond4/test_white_ID{}.png'.format(id_source))
            plt.close()

    elif condition == 5:    #model work in CK dataset
        for target_light in range(11):
            source = load_ck()
            reference = load_light_type(target_light, train=True)
            reference_index = [i for i in range(reference.shape[0])]
            shuffle(reference_index)
            reference_index = reference_index[:5]
            reference = reference[reference_index]

            inputs = tf.concat([source, reference], axis=-1)
            gen_imgs = relight_cycle_generator.predict(tf.concat([source, reference], axis=-1))

            gen_imgs = 0.5 * (gen_imgs + 1)
            source = 0.5 * (source + 1)
            reference = 0.5 * (reference + 1)

            r, c = 3, 5
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0
            for k in range(c):
                axs[0, k].imshow(source[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_imgs[cnt], cmap='gray')
                axs[1, k].axis('off')
                axs[2, k].imshow(reference[cnt], cmap='gray')
                axs[2, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond5/target_light{}.png'.format(target_light))
            plt.close()

    elif condition == 6:    #visualize saliency map
        vsn = build_vsn()
        vsn.load_weights('weight/vsn_weights_21')

        for i in range(32):
            source = load_id(i, train=True)
            source = 0.5 * (source + 1)
            att, _ = vsn.predict(source)
            att = np.mean(att, axis=-1)

            r, c = 2, 11
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(30, 30))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0
            for k in range(c):
                axs[0, k].imshow(source[cnt], cmap='gray')
                axs[0, k].axis('off')
                img = np.reshape(att[cnt], (32,32,1))
                axs[1, k].imshow(cv2.resize(img, (128,128)), cmap='gray')
                axs[1, k].axis('off')
                cnt += 1
            fig.savefig('picture/cond6/ID_{}.png'.format(i))
            plt.close()

    elif condition == 7:
        img_path = os.listdir('affine_lab_face')
        source = []
        for name in img_path:
            img = cv2.imread('affine_lab_face/' + name)
            img = cv2.resize(img, (128,128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
            source.append(img)
        source = np.array(source)
        source = source/127.5 - 1

        natural_roots, _, expression_roots, _ = load_CK(train=True)
        natural_reference = random.sample(natural_roots, 4)
        expression_referencce = random.sample(expression_roots, 4)
        natural_reference_img = load_image(natural_reference)
        expression_reference_img = load_image(expression_referencce)

        inputs_natural = tf.concat([source, natural_reference_img], axis=-1)
        gen_imgs_natural = relight_cycle_generator.predict(inputs_natural)
        inputs_expression = tf.concat([source, expression_reference_img], axis=-1)
        gen_imgs_expression = relight_cycle_generator.predict(inputs_expression)

        source = 0.5 * (source + 1)
        natural_reference_img = 0.5 * (natural_reference_img + 1)
        gen_imgs_natural      = 0.5 * (gen_imgs_natural + 1)
        expression_reference_img = 0.5 * (expression_reference_img + 1)
        gen_imgs_expression      = 0.5 * (gen_imgs_expression + 1)
        r, c = 3, 4
        fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(30, 30))
        plt.subplots_adjust(hspace=0.2)
        cnt = 0
        for k in range(c):
            axs[0, k].imshow(source[cnt], cmap='gray')
            axs[0, k].axis('off')
            axs[1, k].imshow(gen_imgs_natural[cnt], cmap='gray')
            axs[1, k].axis('off')
            axs[2, k].imshow(natural_reference_img[cnt], cmap='gray')
            axs[2, k].axis('off')

            cnt += 1
        fig.savefig('picture/cond7/lab_natural.jpg')
        plt.close(fig)

        r, c = 3, 4
        fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(30, 30))
        plt.subplots_adjust(hspace=0.2)
        cnt = 0
        for k in range(c):
            axs[0, k].imshow(source[cnt], cmap='gray')
            axs[0, k].axis('off')
            axs[1, k].imshow(gen_imgs_expression[cnt], cmap='gray')
            axs[1, k].axis('off')
            axs[2, k].imshow(expression_reference_img[cnt], cmap='gray')
            axs[2, k].axis('off')

            cnt += 1
        fig.savefig('picture/cond7/lab_expression.jpg')

    elif condition == 8:    #random show ck work on testing data
        test_natural_roots, test_id_label_natural, \
        test_expression_roots, test_id_label_expression = load_CK(train=False)
        test_input_image_roots, test_reference_image_roots, test_GT_image_roots, test_id_class \
            = cycle_dataset_ck(test_natural_roots, test_id_label_natural,
                               test_expression_roots, test_id_label_expression)
        for i in range(10):
            source = load_image(get_batch_data(test_input_image_roots, i, 10))
            reference = load_image(get_batch_data(test_reference_image_roots, i, 10))
            gt = load_image(get_batch_data(test_GT_image_roots, i, 10))
            inputs_1 = tf.concat([source, reference], -1)
            gen_imgs_1 = relight_cycle_generator.predict(inputs_1)
            inputs_2 = tf.concat([gen_imgs_1, source], -1)
            gen_imgs_2 = relight_cycle_generator.predict(inputs_2)

            # Rescale images 0 - 1
            source = 0.5 * (source + 1)
            reference = 0.5 * (reference + 1)
            gt = 0.5 * (gt + 1)
            gen_imgs_1 = 0.5 * (gen_imgs_1 + 1)
            gen_imgs_2 = 0.5 * (gen_imgs_2 + 1)
            r, c = 5, 10
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0
            for j in range(c):
                axs[0, j].imshow(source[cnt], cmap='gray')
                axs[0, j].axis('off')
                axs[1, j].imshow(gen_imgs_2[cnt], cmap='gray')
                axs[1, j].axis('off')
                axs[2, j].imshow(gen_imgs_1[cnt], cmap='gray')
                axs[2, j].axis('off')
                axs[3, j].imshow(gt[cnt], cmap='gray')
                axs[3, j].axis('off')
                axs[4, j].imshow(reference[cnt], cmap='gray')
                axs[4, j].axis('off')

                cnt += 1
            fig.savefig('picture/cond8/total_test_{}'.format(i))
            plt.close()

    elif condition == 9:    #light transform total result on testing
        for id_source in range(6):

            for target_light in range(11):
                source = load_id(id_source, train=False)
                reference = load_light_type(target_light, train=False)
                gt = reference[id_source]
                reference = np.delete(reference, [id_source], axis=0)
                start = randint(1, 4)
                reference_index = [(start + i) % 5 for i in range(11)]
                reference_ = reference[reference_index]
                gen_imgs = relight_cycle_generator.predict(tf.concat([source, reference_], axis=-1))
                gen_imgs_2 = relight_cycle_generator.predict(tf.concat([gen_imgs, source], axis=-1))
                gen_imgs = 0.5 * (gen_imgs + 1)
                gen_imgs_2 = 0.5 * (gen_imgs_2 + 1)
                source = 0.5 * (source + 1)
                reference_ = 0.5 * (reference_ + 1)

                r, c = 5, 6
                fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
                plt.subplots_adjust(hspace=0.2)
                cnt = 0
                for k in range(c):
                    axs[0, k].imshow(source[cnt], cmap='gray')
                    axs[0, k].axis('off')
                    axs[1, k].imshow(gen_imgs_2[cnt], cmap='gray')
                    axs[1, k].axis('off')
                    axs[2, k].imshow(gen_imgs[cnt], cmap='gray')
                    axs[2, k].axis('off')
                    axs[3, k].imshow(gt, cmap='gray')
                    axs[3, k].axis('off')
                    axs[4, k].imshow(reference_[cnt] , cmap='gray')
                    axs[4, k].axis('off')

                    cnt += 1
                fig.savefig('picture/cond9/20_final_test_ID{}_light{}.png'.format(id_source, target_light))
                plt.close()

    elif condition == 10:   #final test result on lab photo

        for light_type in range(11):
            img_path = os.listdir('affine_lab_face')
            source = []
            for name in img_path:
                img = cv2.imread('affine_lab_face/' + name)
                img = cv2.resize(img, (128, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=-1)
                source.append(img)
            source = np.array(source)
            source = source / 127.5 - 1

            reference = load_light_type(light_type, train=True)
            reference_index = [i for i in range(len(reference))]
            random.shuffle(reference_index)
            reference = reference[reference_index[:4]]

            gen_imgs = relight_cycle_generator.predict(tf.concat([source, reference], axis=-1))
            gen_imgs_2 = relight_cycle_generator.predict(tf.concat([gen_imgs, source], axis=-1))

            source = 0.5 * (source + 1)
            reference = 0.5 * (reference + 1)
            gen_imgs = 0.5 * (gen_imgs + 1)
            gen_imgs_2 = 0.5 * (gen_imgs_2 + 1)

            r, c = 4, 4
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0
            for k in range(c):
                axs[0, k].imshow(source[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_imgs_2[cnt], cmap='gray')
                axs[1, k].axis('off')
                axs[2, k].imshow(gen_imgs[cnt], cmap='gray')
                axs[2, k].axis('off')
                axs[3, k].imshow(reference[cnt], cmap='gray')
                axs[3, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond10/lab_light{}.png'.format(light_type))
            plt.close()

    elif condition == 11:   #1st: source:ID1 L1, reference: ID2~6 L2~11
        id = 0
        source = load_id(id, train=False)
        for light_type in range(11):
            reference = load_light_type(light_type, train=False)
            reference = np.delete(reference, [id], axis=0)
            source_ = source[light_type]
            source__ = np.zeros_like(reference)
            for i in range(source__.shape[0]):
                source__[i] = source_

            gen_img = relight_cycle_generator.predict(tf.concat([source__, reference], axis=-1))
            r, c = 3, 5
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0

            source__ = 0.5 * (source__ + 1)
            reference = 0.5 * (reference + 1)
            gen_img = 0.5 * (gen_img + 1)
            for k in range(c):

                axs[0, k].imshow(source__[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_img[cnt], cmap='gray')
                axs[1, k].axis('off')
                axs[2, k].imshow(reference[cnt], cmap='gray')
                axs[2, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond11/1_ID{}_light{}.png'.format(id, light_type))
            plt.close()
            #2nd source: ID1 L1, reference:ID1 L2~11
            reference = np.delete(source, [light_type], axis=0)
            source_ = np.zeros_like(reference)
            for i in range(source_.shape[0]):
                source_[i] = source[light_type]
            gen_img = relight_cycle_generator.predict(tf.concat([source_, reference], axis=-1))

            source_ = 0.5 * (source_ + 1)
            reference = 0.5 * (reference + 1)
            gen_img = 0.5 * (gen_img + 1)
            r, c = 3, 5
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0

            source__ = 0.5 * (source__ + 1)
            reference = 0.5 * (reference + 1)
            gen_img = 0.5 * (gen_img + 1)
            for k in range(c):
                axs[0, k].imshow(source_[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_img[cnt], cmap='gray')
                axs[1, k].axis('off')
                axs[2, k].imshow(reference[cnt], cmap='gray')
                axs[2, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond11/2_ID{}_light{}.png'.format(id, light_type))
            plt.close()

    elif condition == 12:   #all id in ck testing data N2E
        path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK/test'
        #N -> E
        for id_ in range(33):
            source_root = []
            reference_root = []
            natural_path = path + '/Natural image'
            natural_file = os.listdir(natural_path) #S001, S002, ...
            natural_file.sort()
            natural_id_file = natural_path + '/' + natural_file[id_]     #id_th ppl od natural expression
            natural_id_file = natural_id_file + '/' + os.listdir(natural_id_file)[0]
            natural_id_img_name = os.listdir(natural_id_file)
            for img_name in natural_id_img_name:
                source_root.append(natural_id_file + '/' + img_name)

            expression_path = path + '/Expression image'
            expression_file = os.listdir(expression_path)
            expression_file.sort()
            del expression_file[id_]
            for other_id in expression_file:
                expression_id_file = expression_path + '/' + other_id
                expression_id_file = expression_id_file + '/' + os.listdir(expression_id_file)[0]
                expression_id_img_name = os.listdir(expression_id_file)
                for img_name in expression_id_img_name:
                    reference_root.append(expression_id_file + '/' + img_name)

            random.shuffle(reference_root)
            reference_root = reference_root[:len(source_root)]

            source = load_image(source_root)
            reference = load_image(reference_root)
            gen_img1 = relight_cycle_generator.predict(tf.concat([source, reference], -1))
            gen_img2 = relight_cycle_generator.predict(tf.concat([gen_img1, source], -1))

            source = 0.5 * (source + 1)
            reference = 0.5 * (reference + 1)
            gen_img1 = 0.5 * (gen_img1 + 1)
            gen_img2 = 0.5 * (gen_img2 + 1)

            r, c = 4, source.shape[0]
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0

            for k in range(c):
                axs[0, k].imshow(source[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_img2[cnt], cmap='gray')
                axs[1, k].axis('off')
                axs[2, k].imshow(gen_img1[cnt], cmap='gray')
                axs[2, k].axis('off')
                axs[3, k].imshow(reference[cnt], cmap='gray')
                axs[3, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond12/{}_N2E.png'.format(natural_file[id_]))
            plt.close()

    elif condition == 13:   #all id in ck testing data E2N
        path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK/test'
        #N -> E
        for id_ in range(33):
            source_root = []
            reference_root = []
            expression_path = path + '/Expression image'
            expression_file = os.listdir(expression_path) #S001, S002, ...
            expression_file.sort()
            expression_id_file = expression_path + '/' + expression_file[id_]     #id_th ppl od natural expression
            expression_id_file = expression_id_file + '/' + os.listdir(expression_id_file)[0]
            expression_id_img_name = os.listdir(expression_id_file)
            for img_name in expression_id_img_name:
                source_root.append(expression_id_file + '/' + img_name)

            natural_path = path + '/Natural image'
            natural_file = os.listdir(natural_path)
            natural_file.sort()
            del natural_file[id_]
            for other_id in natural_file:
                natural_id_file = natural_path + '/' + other_id
                natural_id_file = natural_id_file + '/' + os.listdir(natural_id_file)[0]
                natural_id_img_name = os.listdir(natural_id_file)
                for img_name in natural_id_img_name:
                    reference_root.append(natural_id_file + '/' + img_name)

            random.shuffle(reference_root)
            reference_root = reference_root[:len(source_root)]

            source = load_image(source_root)
            reference = load_image(reference_root)
            gen_img1 = relight_cycle_generator.predict(tf.concat([source, reference], -1))
            gen_img2 = relight_cycle_generator.predict(tf.concat([gen_img1, source], -1))

            source = 0.5 * (source + 1)
            reference = 0.5 * (reference + 1)
            gen_img1 = 0.5 * (gen_img1 + 1)
            gen_img2 = 0.5 * (gen_img2 + 1)

            r, c = 4, source.shape[0]
            fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0

            for k in range(c):
                axs[0, k].imshow(source[cnt], cmap='gray')
                axs[0, k].axis('off')
                axs[1, k].imshow(gen_img2[cnt], cmap='gray')
                axs[1, k].axis('off')
                axs[2, k].imshow(gen_img1[cnt], cmap='gray')
                axs[2, k].axis('off')
                axs[3, k].imshow(reference[cnt], cmap='gray')
                axs[3, k].axis('off')

                cnt += 1
            fig.savefig('picture/cond13/{}_E2N.png'.format(expression_file[id_]))
            plt.close()
