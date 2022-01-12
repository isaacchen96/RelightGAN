from build_model import *
from load_data import *
from train import *
import numpy as np
import time
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt

class Relight_cycle_pretrain:
    def __init__(self):
        self.generator = build_relight()
        self.discriminator = build_discriminator()
        self.opt = Adam(1e-4)
        self.style_l = style_loss()
        self.path = '/home/pomelo96/Desktop/datasets/Yaleb/train'
        self.train_roots, self.train_id, self.train_light, _, _, _ = load_YaleB()
        self.input_image_roots, self.reference_image_roots, self.GT_image_roots, self.id_class \
            = set_data_for_cycleGAN(self.train_roots, self.train_light, is_pretrain=True)
        self.source_sampling = load_image(get_batch_data(self.input_image_roots, 0, 10))
        self.reference_sampling = load_image(get_batch_data(self.reference_image_roots, 0, 10))
        self.gt_sampling = load_image(get_batch_data(self.GT_image_roots, 0, 10))

    def gen_train_step(self, source, reference, label):
        label = tf.one_hot(label, depth=32)
        with tf.GradientTape() as tape:
            inputs = tf.concat([source, reference], axis=-1)
            gen_img = self.generator.call(inputs)
            v_gen, c_gen = self.discriminator.call(gen_img)
            loss_cls = classify_loss(label, c_gen)
            loss_adv = adversarial_loss(target = True, pred = v_gen)
            loss_img = img_loss(source, gen_img)
            loss_style = self.style_l.predict_loss(reference, gen_img)
            loss_g = tf.reduce_mean([loss_cls, loss_adv, loss_img, loss_style])
        grads = tape.gradient(loss_g, self.generator.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.generator.trainable_variables))

        return loss_g, loss_adv, loss_img

    def dis_train_step(self, source, reference, label):
        label = tf.one_hot(label, depth=32)
        with tf.GradientTape() as tape:
            inputs = tf.concat([source, reference], axis=-1)
            gen_img = self.generator.call(inputs)
            v_gen, c_gen = self.discriminator.call(gen_img)
            v_real, c_real = self.discriminator.call(tf.cast(source, dtype='float32'))
            # loss_classify_gen = classify_loss(label, c_gen)
            #This will make generator's predict all become reference and even id will predicted correctly
            #AI magic wtf lol lmao kaobei om gash
            loss_classify_real = classify_loss(label, c_real)
            loss_adv_gen = adversarial_loss(target=False, pred=v_gen)
            loss_adv_real = adversarial_loss(target=True, pred=v_real)

            loss_d = tf.reduce_mean([loss_classify_real, loss_adv_gen, loss_adv_real])
        grads = tape.gradient(loss_d, self.discriminator.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return  loss_d, np.mean([loss_adv_gen, loss_adv_real]), np.mean([loss_classify_real])

    def pretrain(self, epochs=200, interval=1, batch_size=32, batch_num=341):
        tr_L_G_avg = []
        tr_L_G_adv_avg = []
        tr_L_G_img_avg = []
        tr_L_D_avg = []
        tr_L_D_adv_avg = []
        tr_L_D_cls_avg = []
        start = time.time()
        for epoch in range(epochs):
            ep_start = time.time()
            tr_L_G = []
            tr_L_G_adv = []
            tr_L_G_img = []
            tr_L_D = []
            tr_L_D_adv = []
            tr_L_D_cls = []

            for b in range(batch_num):
                source = load_image(get_batch_data(self.input_image_roots, b, batch_size))
                reference = load_image(get_batch_data(self.reference_image_roots, b, batch_size))
                target = load_image(get_batch_data(self.GT_image_roots, b, batch_size))
                label = get_batch_data(self.id_class, b, batch_size)
                loss_g, loss_adv_g, loss_img_g = self.gen_train_step(source, reference, label)
                tr_L_G.append(loss_g)
                tr_L_G_adv.append(loss_adv_g)
                tr_L_G_img.append(loss_img_g)
                loss_d, loss_adv_d, loss_cls_d = self.dis_train_step(source, reference, label)
                tr_L_D.append(loss_d)
                tr_L_D_adv.append(loss_adv_d)
                tr_L_D_cls.append(loss_cls_d)
            tr_L_G_avg.append(np.mean(tr_L_G))
            tr_L_G_adv_avg.append(np.mean(tr_L_G_adv))
            tr_L_G_img_avg.append(np.mean(tr_L_G_img))
            tr_L_D_avg.append(np.mean(tr_L_D))
            tr_L_D_adv_avg.append(np.mean(tr_L_D_adv))
            tr_L_D_cls_avg.append(np.mean(tr_L_D_cls))

            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)
            print('\nTime for pass  {:<4d}: {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                        int(m_pass), s_pass))
            print('Time for epoch {:<4d}: {:6.3f} sec'.format(epoch + 1, time.time() - ep_start))
            print('Train Loss Gen_adv       :  {:8.5f}'.format(tr_L_G_adv_avg[-1]))
            print('Train Loss Dis_adv       :  {:8.5f}'.format(tr_L_D_adv_avg[-1]))
            print('Train Loss Generator     :  {:8.5f}'.format(tr_L_G_avg[-1]))
            print('Train Loss Gen img       :  {:8.5f}'.format(tr_L_G_img_avg[-1]))
            print('Train Loss Discriminator :  {:8.5f}'.format(tr_L_D_avg[-1]))
            print('Train Loss Dis class     :  {:8.5f}'.format(tr_L_D_cls_avg[-1]))

            if epoch % interval == 0 or epoch + 1 == epochs:
                self.sample_images_pretrain(epoch, self.source_sampling, self.reference_sampling, self.gt_sampling)
                self.generator.save_weights('pretrain_weight/generator_pretrained_weights_{}'.format(epoch+1))
                self.discriminator.save_weights('pretrain_weight/discriminator_pretrained_weights_{}'.format(epoch+1))
        return tr_L_G_avg, tr_L_D_avg, tr_L_G_adv_avg, tr_L_D_adv_avg, tr_L_G_img_avg, tr_L_D_cls_avg

    def sample_images_pretrain(self, epoch, source, reference, gt):

        inputs_ = tf.concat([source, reference], -1)
        gen_imgs = self.generator.predict(inputs_)
        # Rescale images 0 - 1
        source = 0.5 * (source + 1)
        reference = 0.5 * (reference + 1)
        gt = 0.5 * (gt + 1)
        gen_imgs = 0.5 * (gen_imgs + 1)
        r, c = 4, 10
        fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
        plt.subplots_adjust(hspace=0.2)
        cnt = 0
        for j in range(c):
            axs[0, j].imshow(source[cnt], cmap='gray')
            axs[0, j].axis('off')
            axs[1, j].imshow(gen_imgs[cnt], cmap='gray')
            axs[1, j].axis('off')
            axs[2, j].imshow(gt[cnt], cmap='gray')
            axs[2, j].axis('off')
            axs[3, j].imshow(reference[cnt], cmap='gray')
            axs[3, j].axis('off')

            cnt += 1
        fig.savefig('pretrain_picture/pretrain_{}.png'.format(epoch+1))
        plt.close()

if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config = ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    relight_cycle = Relight_cycle_pretrain()
    # relight_cycle.generator.load_weights('pretrain_weight/generator_pretrained_weights_10')
    # relight_cycle.discriminator.load_weights('pretrain_weight/discriminator_pretrained_weights_10')
    tr_L_G_avg, tr_L_D_avg, tr_L_G_adv_avg, tr_L_D_adv_avg, tr_L_G_img_avg, tr_L_D_cls_avg = relight_cycle.pretrain(epochs=20, interval=1)

    plt.plot(tr_L_G_avg)
    plt.plot(tr_L_D_avg)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('Pretrain Generator Loss')
    plt.savefig('pretrain_picture/pretrain_loss.jpg')
    plt.close()

    plt.plot(tr_L_G_adv_avg)
    plt.plot(tr_L_D_adv_avg)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('Pretrain Adversarial Loss')
    plt.savefig('pretrain_picture/pretrain_Adversarial_loss.jpg')
    plt.close()

    plt.plot(tr_L_G_img_avg)
    plt.legend(['Image loss'])
    plt.savefig('pretrain_picture/pretrain_imgae_loss.jpg')
    plt.close()

    plt.plot(tr_L_D_cls_avg)
    plt.legend(['Classify loss'])
    plt.savefig('pretrain_picture/pretrain_classify_loss.jpg')
    plt.close()
