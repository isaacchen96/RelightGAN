import time
from build_model import *
from load_data_path import *
from loss_function import *
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import accuracy_score

class Relight_cycle:
    def __init__(self):
        self.generator = build_relight()
        self.discriminator = build_discriminator()
        self.vsn = build_vsn()
        self.opt = Adam(1e-4)
        self.train_roots, self.train_id, self.train_light, self.test_roots, self.test_id, self.test_light = load_YaleB()
        self.input_image_roots, self.reference_image_roots, self.GT_image_roots, self.id_class \
            = set_data_for_cycleGAN(self.train_roots, self.train_light, is_pretrain=False)
        self.test_input_image_roots, self.test_reference_image_roots, self.test_GT_image_roots, self.test_id_class \
            = set_data_for_cycleGAN(self.test_roots, self.test_light, is_pretrain=False)
        self.style_l = style_loss()

    def gen_train_step(self, source, reference, target, label, training = True):
        label = tf.one_hot(label, depth=46)
        with tf.GradientTape() as tape:
            inputs = tf.concat([source, reference], -1)
            gen_img = self.generator.call(inputs)
            v_gen, c_gen = self.discriminator.call(gen_img)
            loss_classify = classify_loss(label, c_gen)
            loss_img     = img_loss(target, gen_img)
            loss_style   = self.style_l.predict_loss(reference, gen_img)
            loss_adv     = adversarial_loss(target=True, pred=v_gen)
            loss_g = loss_img + loss_style + loss_adv + loss_classify
        if training:
            loss_g = loss_img + loss_style + loss_adv + loss_classify
            grads = tape.gradient(loss_g, self.generator.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        else:
            loss_g = loss_img + loss_style + loss_adv

        return loss_g, loss_adv, loss_img, loss_style, loss_classify

    def dis_train_step(self, source, reference, target, label, training = True):
        label = tf.one_hot(label, depth=46)
        with tf.GradientTape() as tape:
            inputs = tf.concat([source, reference], -1)
            gen_img = self.generator.call(inputs)
            v_gen, c_gen = self.discriminator.call(gen_img)
            v_real, c_real = self.discriminator.call(tf.cast(target, dtype='float32'))
            #loss_classify_gen = classify_loss(label, c_gen)
            loss_classify_real = classify_loss(label, c_real)
            loss_adv_gen = adversarial_loss(target=False, pred=v_gen)
            loss_adv_real = adversarial_loss(target=True, pred=v_real)
            loss_adv = 0.5 * (loss_adv_gen + loss_adv_real)
            loss_d = loss_classify_real + loss_adv
        if training:
            grads = tape.gradient(loss_d, self.discriminator.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        else:
            loss_d = loss_adv
        return loss_d,  loss_adv, loss_classify_real

    def train(self, epochs=50, interval=1, batch_size=17, batch_num=440):
        tr_L_G_avg = []
        tr_L_G_adv_avg = []
        tr_L_G_img_avg = []
        tr_L_G_style_avg = []
        tr_L_G_cls_avg = []
        tr_L_D_avg = []
        tr_L_D_adv_avg=[]
        tr_L_D_cls_avg = []
        te_L_G_avg = []
        te_L_D_avg = []
        start = time.time()
        for epoch in range(epochs):
            tr_L_G = []
            tr_L_G_adv = []
            tr_L_G_img = []
            tr_L_G_style = []
            tr_L_G_cls = []
            tr_L_D = []
            tr_L_D_adv = []
            tr_L_D_cls = []
            te_L_G = []
            te_L_D = []

            ep_start = time.time()
            for b in range(batch_num):
                source = load_image(get_batch_data(self.input_image_roots, b, batch_size))
                reference = load_image(get_batch_data(self.reference_image_roots, b, batch_size))
                target = load_image(get_batch_data(self.GT_image_roots, b, batch_size))
                label = get_batch_data(self.id_class, b, batch_size)

                b_test = randint(0,80)
                test_source = load_image(get_batch_data(self.test_input_image_roots, b_test, batch_size))
                test_reference = load_image(get_batch_data(self.test_reference_image_roots, b_test, batch_size))
                test_target = load_image(get_batch_data(self.test_GT_image_roots, b_test, batch_size))
                test_label = get_batch_data(self.test_id_class, b_test, batch_size)

                for i in range(2):
                    loss_g, loss_adv, loss_img, loss_style, loss_cls_g \
                            = self.gen_train_step(source, reference, target, label)
                loss_g_test, _, _, _, _ = self.gen_train_step(test_source, test_reference, test_target, test_label, training=False)
                tr_L_G.append(loss_g)
                tr_L_G_adv.append(loss_adv)
                tr_L_G_img.append(loss_img)
                tr_L_G_style.append(loss_style)
                tr_L_G_cls.append(loss_cls_g)
                te_L_G.append(loss_g_test)

                loss_d, loss_adv_d, loss_cls_d = self.dis_train_step(source, reference, target, label)
                loss_d_test, _, _ = self.dis_train_step(test_source, test_reference, test_target, test_label, training=False)
                tr_L_D.append(loss_d)
                tr_L_D_adv.append(loss_adv_d)
                tr_L_D_cls.append(loss_cls_d)
                te_L_D.append(loss_d_test)
                source, reference, target, label = None, None, None, None

            tr_L_G_avg.append(np.mean(tr_L_G))
            tr_L_G_adv_avg.append(np.mean(tr_L_G_adv))
            tr_L_G_img_avg.append(np.mean(tr_L_G_img))
            tr_L_G_style_avg.append(np.mean(tr_L_G_style))
            tr_L_G_cls_avg.append(np.mean(tr_L_G_cls))
            tr_L_D_avg.append(np.mean(tr_L_D))
            tr_L_D_adv_avg.append(np.mean(tr_L_D_adv))
            tr_L_D_cls_avg.append(np.mean(tr_L_D_cls))
            te_L_G_avg.append(np.mean(te_L_G))
            te_L_D_avg.append(np.mean(te_L_D))

            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)
            print('\nTime for pass  {:<4d}  : {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                      int(m_pass), s_pass))
            print('Time for epoch {:<4d}    : {:6.3f} sec'.format(epoch + 1, time.time() - ep_start))
            print('Train Loss Gen_adv       :  {:8.5f}'.format(tr_L_G_adv_avg[-1]))
            print('Train Loss Gen_img       :  {:8.5f}'.format(tr_L_G_img_avg[-1]))
            print('Train Loss Gen_style     :  {:8.5f}'.format(tr_L_G_style_avg[-1]))
            print('Train Loss Gen_classify  :  {:8.5f}'.format(tr_L_G_cls_avg[-1]))
            print('Train Loss Generator     :  {:8.5f}'.format(tr_L_G_avg[-1]))
            print('Train Loss Dis_cls       :  {:8.5f}'.format(tr_L_D_cls_avg[-1]))
            print('Train Loss Dis_adv       :  {:8.5f}'.format(tr_L_D_adv_avg[-1]))
            print('Train Loss Discriminator :  {:8.5f}'.format(tr_L_D_avg[-1]))
            print('Test Loss Generator      :  {:8.5f}'.format(te_L_G_avg[-1]))
            print('Test Loss Discriminator  :  {:8.5f}'.format(te_L_D_avg[-1]))
            self.sample_images(epoch, path = 'picture2/total_train')
            if (epoch % interval == 0 or epoch + 1 == epochs) and (te_L_G_avg[-1] <= np.min(te_L_G_avg)):
                self.generator.save_weights('weight/generator_weights_{}'.format(epoch+1))
                self.discriminator.save_weights('weight/discriminator_weights_{}'.format(epoch+1))
        return [tr_L_G_avg, tr_L_G_adv_avg, tr_L_G_img_avg, tr_L_G_style_avg, tr_L_G_cls_avg],\
               [tr_L_D_avg, tr_L_D_adv_avg, tr_L_D_cls_avg], [te_L_G_avg, te_L_D_avg]

    def train_vsn_step(self, source, light_label, training=True):
        light_label = tf.one_hot(light_label, depth=2)
        with tf.GradientTape() as tape:
            _, pred = self.vsn(source)
            loss_cls = classify_loss(light_label, pred)
            acc = accuracy_score(np.argmax(light_label, axis=-1), np.argmax(pred, axis=-1))
        if training:
            grads = tape.gradient(loss_cls, self.vsn.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.vsn.trainable_variables))

        return loss_cls, acc

    def train_vsn(self, epochs=500, interval=5, batch_size=11, batch_num=32):
        tr_L_vsn_avg = []
        tr_acc_vsn_avg = []
        te_L_vsn_avg = []
        te_acc_vsn_avg = []
        start = time.time()

        for epoch in range(epochs):
            ep_start = time.time()
            tr_L_vsn = []
            tr_acc_vsn = []
            train_roots = np.array(self.train_roots)
            train_light = np.array(self.train_light)
            train_idx = [i for i in range(batch_size*batch_num)]
            random.shuffle(train_idx)
            for b in range(batch_num):
                idx = train_idx[b*batch_size: (b+1)*batch_size]
                source = load_image(train_roots[idx])
                source = 0.5 * (source + 1)
                label = train_light[idx]
                loss_vsn, tr_acc = self.train_vsn_step(source, label, training=True)
                tr_L_vsn.append(loss_vsn)
                tr_acc_vsn.append(tr_acc)
            tr_L_vsn_avg.append(np.mean(tr_L_vsn))
            tr_acc_vsn_avg.append(np.mean(tr_acc))
            test_img = load_image(self.test_roots)
            test_img = 0.5 * (test_img + 1)
            test_light = self.test_light
            loss_cls_test, te_acc = self.train_vsn_step(test_img, test_light, training=False)
            te_L_vsn_avg.append(loss_cls_test)
            te_acc_vsn_avg.append(te_acc)

            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)
            print('\nTime for pass  {:<4d}  : {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                          int(m_pass), s_pass))
            print('Time for epoch {:<4d}    : {:6.3f} sec'.format(epoch + 1, time.time() - ep_start))
            print('Train Loss VSN           :  {:8.5f}'.format(tr_L_vsn_avg[-1]))
            print('Train accuracy           :  {:8.5f}'.format(tr_acc_vsn_avg[-1]))
            print('Test Loss VSN            :  {:8.5f}'.format(te_L_vsn_avg[-1]))
            print('Test accuracy            :  {:8.5f}'.format(te_acc_vsn_avg[-1]))

            if (epoch % interval == 0 or epoch + 1 == epochs) and (te_L_vsn_avg[-1] <= np.min(te_L_vsn_avg)):
                self.vsn.save_weights('weight/vsn_weights_{}'.format(epoch + 1))

        return tr_L_vsn_avg, tr_acc_vsn_avg, te_L_vsn_avg, te_acc_vsn_avg

    def gen_train_step_part2(self, source, reference, target, label, training=True):
        label = tf.one_hot(label, depth=46)
        with tf.GradientTape() as tape:
            inputs = tf.concat([source, reference], -1)
            gen_img_1 = self.generator.call(inputs)
            v_gen, c_gen = self.discriminator.call(gen_img_1)
            inputs_2 = tf.concat([gen_img_1, source], -1)
            gen_img_2 = self.generator.call(inputs_2)
            att, _ = self.vsn(source)
            att = tf.reduce_mean(att, axis=-1)
            att = tf.reshape(att, (att.shape[0], att.shape[1], att.shape[2], 1))

            loss_classify = classify_loss(label, c_gen)
            loss_img = img_loss(target, gen_img_1)
            loss_style = self.style_l.predict_loss(reference, gen_img_1)
            loss_adv = adversarial_loss(target=True, pred=v_gen)
            loss_psnr = PSNR_loss(source, gen_img_2)
            loss_ssim = SSIM_loss(source, gen_img_2)
            loss_ssim = tf.reduce_mean(loss_ssim)
            loss_ssim_att = SSIM_att_loss(source, gen_img_2, att)
            loss_ssim_att = tf.reduce_mean(loss_ssim_att)
            loss_cycle = img_loss(source, gen_img_2)
            loss_g = loss_adv + loss_classify + 2 * loss_img + loss_style + loss_psnr + 2 * loss_ssim + 4 * loss_ssim_att + 10 * loss_cycle
        if training:
            grads = tape.gradient(loss_g, self.generator.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        else:
            loss_g = loss_adv + 2 * loss_img + loss_style + loss_psnr + 2 * loss_ssim + 4 * loss_ssim_att + 10 * loss_cycle

        return loss_g, loss_adv, loss_classify, 2 * loss_img, loss_style, loss_psnr, 2 * loss_ssim, 4 * loss_ssim_att, 10 * loss_cycle

    def train_part2(self, epochs=30, interval=1, batch_size=16, train_num=880):
        tr_L_G_avg, tr_L_G_adv_avg, tr_L_G_cls_avg, tr_L_G_img_avg, tr_L_G_style_avg = [], [], [], [], []
        tr_L_G_psnr_avg, tr_L_G_ssim_avg, tr_L_G_ssim_att_avg = [], [], []
        tr_L_G_cycle_avg, tr_L_D_avg, tr_L_D_adv_avg, tr_L_D_cls_avg = [], [], [], []
        te_L_G_avg, te_L_D_avg = [], []
        start = time.time()
        for epoch in range(epochs):
            tr_L_G, tr_L_G_adv, tr_L_G_cls, tr_L_G_img, tr_L_G_style = [], [], [], [], []
            tr_L_G_psnr, tr_L_G_ssim, tr_L_G_ssim_att = [], [], []
            tr_L_G_cycle, tr_L_D, tr_L_D_adv, tr_L_D_cls = [], [], [], []
            te_L_G, te_L_D = [], []

            ep_start = time.time()
            for b in range(train_num):
                source = load_image(get_batch_data(self.input_image_roots, b, batch_size))
                reference = load_image(get_batch_data(self.reference_image_roots, b, batch_size))
                target = load_image(get_batch_data(self.GT_image_roots, b, batch_size))
                label = get_batch_data(self.id_class, b, batch_size)

                b_test = randint(0, 80)
                test_source = load_image(get_batch_data(self.test_input_image_roots, b_test, batch_size))
                test_reference = load_image(get_batch_data(self.test_reference_image_roots, b_test, batch_size))
                test_target = load_image(get_batch_data(self.test_GT_image_roots, b_test, batch_size))
                test_label = get_batch_data(self.test_id_class, b_test, batch_size)

                loss_g, loss_adv_g, loss_cls_g, loss_img, loss_style, loss_psnr, loss_ssim, loss_ssim_att, loss_cycle = \
                   self.gen_train_step_part2(source, reference, target, label)

                loss_g_test, _, _, _, _, _, _, _, _ = self.gen_train_step_part2(test_source, test_reference, test_target, test_label,
                                                              training=False)
                loss_d, loss_adv_d, loss_cls_d = self.dis_train_step(source, reference, target, label)
                loss_d_test, _, _ = self.dis_train_step(test_source, test_reference, test_target, test_label,
                                                        training=False)

                tr_L_G.append(loss_g)
                tr_L_G_adv.append(loss_adv_g)
                tr_L_G_cls.append(loss_cls_g)
                tr_L_G_img.append(loss_img)
                tr_L_G_style.append(loss_style)
                tr_L_G_psnr.append(loss_psnr)
                tr_L_G_ssim.append(loss_ssim)
                tr_L_G_ssim_att.append(loss_ssim_att)
                tr_L_G_cycle.append(loss_cycle)
                tr_L_D.append(loss_d)
                tr_L_D_adv.append(loss_adv_d)
                tr_L_D_cls.append(loss_cls_d)
                te_L_G.append(loss_g_test)
                te_L_D.append(loss_d_test)

            tr_L_G_avg.append(np.mean(tr_L_G))
            tr_L_G_cls_avg.append(np.mean(tr_L_G_cls))
            tr_L_G_img_avg.append(np.mean(tr_L_G_img))
            tr_L_G_style_avg.append(np.mean(tr_L_G_style))
            tr_L_G_adv_avg.append(np.mean(tr_L_G_adv))
            tr_L_G_psnr_avg.append(np.mean(tr_L_G_psnr))
            tr_L_G_ssim_avg.append(np.mean(tr_L_G_ssim))
            tr_L_G_ssim_att_avg.append(np.mean(tr_L_G_ssim_att))
            tr_L_G_cycle_avg.append(np.mean(tr_L_G_cycle))
            tr_L_D_avg.append(np.mean(tr_L_D))
            tr_L_D_adv_avg.append(np.mean(tr_L_D_adv))
            tr_L_D_cls_avg.append(np.mean(tr_L_D_cls))
            te_L_G_avg.append(np.mean(te_L_G))
            te_L_D_avg.append(np.mean(te_L_D))

            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)
            print('\nTime for pass  {:<4d}  : {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                          int(m_pass), s_pass))
            print('Time for epoch {:<4d}    : {:6.3f} sec'.format(epoch + 1, time.time() - ep_start))
            print('Train Loss Generator     :  {:8.5f}'.format(tr_L_G_avg[-1]))
            print('Train Loss Gen_adv       :  {:8.5f}'.format(tr_L_G_adv_avg[-1]))
            print('Train Loss Gen_classify  :  {:8.5f}'.format(tr_L_G_cls_avg[-1]))
            print('Train Loss Gen_img       :  {:8.5f}'.format(tr_L_G_img_avg[-1]))
            print('Train Loss Gen_style     :  {:8.5f}'.format(tr_L_G_style_avg[-1]))
            print('Train Loss Gen_PSNR      :  {:8.5f}'.format(tr_L_G_psnr_avg[-1]))
            print('Train Loss Gen_SSIM      :  {:8.5f}'.format(tr_L_G_ssim_avg[-1]))
            print('Train Loss Gen_SSIM_att  :  {:8.5f}'.format(tr_L_G_ssim_att_avg[-1]))
            print('Train Loss Gen_cycle     :  {:8.5f}'.format(tr_L_G_cycle_avg[-1]))
            print('Train Loss Dis_cls       :  {:8.5f}'.format(tr_L_D_cls_avg[-1]))
            print('Train Loss Dis_adv       :  {:8.5f}'.format(tr_L_D_adv_avg[-1]))
            print('Train Loss Discriminator :  {:8.5f}'.format(tr_L_D_avg[-1]))
            print('Test Loss Generator      :  {:8.5f}'.format(te_L_G_avg[-1]))
            print('Test Loss Discriminator  :  {:8.5f}'.format(te_L_D_avg[-1]))

            self.sample_images(epoch, path = 'picture2/total_train')
            if (epoch % interval == 0 or epoch + 1 == epochs) and (te_L_G_avg[-1] <= np.min(te_L_G_avg)):
                self.generator.save_weights('weight2/generator_part2_weights_{}'.format(epoch + 1))
                self.discriminator.save_weights('weight2/discriminator_part2_weights_{}'.format(epoch + 1))
        return [tr_L_G_avg, tr_L_G_adv_avg, tr_L_G_cls_avg, tr_L_G_img_avg, tr_L_G_style_avg,
                tr_L_G_psnr_avg, tr_L_G_ssim_avg, tr_L_G_ssim_att_avg, tr_L_G_cycle_avg], \
                [tr_L_D_avg, tr_L_D_adv_avg, tr_L_D_cls_avg], [te_L_G_avg, te_L_D_avg]


    def sample_images(self, epoch, path):
        source = load_image(get_batch_data(self.input_image_roots, 0, 10))
        reference = load_image(get_batch_data(self.reference_image_roots, 0, 10))
        gt = load_image(get_batch_data(self.GT_image_roots, 0, 10))
        inputs_1 = tf.concat([source, reference], -1)
        gen_imgs_1 = self.generator.predict(inputs_1)
        inputs_2 = tf.concat([gen_imgs_1, source], -1)
        gen_imgs_2 = self.generator.predict(inputs_2)

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
        fig.savefig(path +'_{}.png'.format(epoch+1))
        plt.close()

if __name__ == '__main__':
    print(tf.__version__)
    print(tf.test.is_gpu_available())
    print(tf.config.list_physical_devices('GPU'))
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config = ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    relight_cycle = Relight_cycle()

    relight_cycle.generator.load_weights('weight/generator_weights_25')
    relight_cycle.discriminator.load_weights('weight/discriminator_weights_25')
    relight_cycle.vsn.load_weights('weight/vsn_weights_50')
    [tr_L_G_avg, tr_L_G_adv_avg, tr_L_G_cls_avg, tr_L_G_img_avg, tr_L_G_style_avg,
     tr_L_G_psnr_avg, tr_L_G_ssim_avg, tr_L_G_ssim_att_avg, tr_L_G_cycle_avg], \
    [tr_L_D_avg, tr_L_D_adv_avg, tr_L_D_cls_avg], [te_L_G_avg, te_L_D_avg] \
        = relight_cycle.train_part2(epochs=50, interval=1)

    plt.plot(tr_L_G_avg)
    plt.title('Generator total loss')
    plt.savefig('picture2/_Generator loss.jpg')
    plt.close()

    plt.plot(tr_L_G_adv_avg)
    plt.plot(tr_L_D_adv_avg)
    plt.title('Adversarial loss')
    plt.legend(['Generator', 'Discriminator'], loc='upper right')
    plt.savefig('picture2/_Adversarial loss.jpg')
    plt.close()

    plt.plot(tr_L_G_img_avg)
    plt.title('Generator Image Loss')
    plt.savefig('picture2/_Generator Image loss.jpg')
    plt.close()

    plt.plot(tr_L_G_style_avg)
    plt.title('Generator style Loss')
    plt.savefig('picture2/_Generator style loss.jpg')
    plt.close()

    plt.plot(tr_L_G_cls_avg)
    plt.title('Generator Classify Loss')
    plt.savefig('picture2/_Generator Classify loss.jpg')
    plt.close()

    plt.plot(tr_L_G_psnr_avg)
    plt.title('Generator PSNR loss')
    plt.savefig('picture2/_Generator part2 PSNR loss.jpg')
    plt.close()

    plt.plot(tr_L_G_ssim_avg)
    plt.title('Generator SSIM loss')
    plt.savefig('picture2/_Generator part2 SSIM loss.jpg')
    plt.close()

    plt.plot(tr_L_G_ssim_att_avg)
    plt.title('Generator SSIM att loss')
    plt.savefig('picture2/_Generator part2 SSIM att loss.jpg')
    plt.close()

    plt.plot(tr_L_G_cycle_avg)
    plt.title('Generator cycle loss')
    plt.savefig('picture2/_Generator part2 cycle loss.jpg')
    plt.close()

    plt.plot(tr_L_D_cls_avg)
    plt.title('Discriminator Classify Loss')
    plt.savefig('picture2/_Discriminator Classify loss.jpg')
    plt.close()

    plt.plot(tr_L_D_avg)
    plt.title('Discriminator total loss')
    plt.savefig('picture2/_Discriminator loss')
    plt.close()

    plt.plot(te_L_G_avg)
    plt.title('Generator test loss')
    plt.savefig('picture2/_Generator part2 test loss.jpg')
    plt.close()

    plt.plot(te_L_D_avg)
    plt.title('Discriminator test loss')
    plt.savefig('picture2/_Discriminator part2 test loss')
    plt.close()

    '''
   tr_L_vsn_avg, tr_acc_vsn_avg, te_L_vsn_avg, te_acc_vsn_avg = relight_cycle.train_vsn(epochs=50, interval=1)
   plt.plot(tr_L_vsn_avg)
   plt.title('Train vsn loss')
   plt.savefig('picture/_train vsn loss.jpg')
   plt.close()

   plt.plot(te_L_vsn_avg)
   plt.title('Test vsn loss')
   plt.savefig('picture/_test vsn loss.jpg')
   plt.close()

   plt.plot(tr_acc_vsn_avg)
   plt.plot(te_acc_vsn_avg)
   plt.title('VSN classify accuracy')
   plt.legend(['Train', 'Test'])
   plt.savefig('picture/_vsn acc.jpg')
   plt.close()
   '''
