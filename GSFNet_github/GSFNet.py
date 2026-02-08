import os
import glob
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from fem import UBM
from gam import BAM
from ffm import PSM
from ssm import GLFormer as GLF
from fm import Fusion
from data_reader import load_batch
from scheduler import schedule
from loss_functions import SmoothL1Loss


from scipy import stats



class BGANet:
    def __init__(self, height=1024, width=1024, channel=3, max_disp=96, num_class=6, filters=16):
        self.height = height
        self.width = width
        self.channel = channel
        self.max_disp = max_disp
        self.num_class = num_class
        self.filters = filters
        self.model = None

    def build_model(self):
        # inputs
        left_image = keras.Input((self.height, self.width, self.channel))
        right_image = keras.Input((self.height, self.width, self.channel))


        # unified backbone module
        backbone = UBM(filters=self.filters)
        left = backbone(left_image,training=True)
        right = backbone(right_image,training=True)
        print('ubm运行完成')

        # bidirectional guided attention module
        attention = BAM(filters=self.filters, dsps=self.max_disp // 4)
        [cls_att, dsp_att] = attention(left)
        print('bam运行完成')

        # semantic segmentation module
        segmentation = GLF(filters=self.filters, num_classes=self.num_class)
        init_score_map = segmentation([left_image,cls_att])
        init_seg_map = tf.argmax(init_score_map, -1)
        init_seg_map = tf.cast(init_seg_map, tf.float32)
        init_seg_map = tf.expand_dims(init_seg_map, -1)
        print('ssm运行完成 ')

        # feature matching module
        match = PSM(filters=self.filters, max_disp=self.max_disp)
        init_dsp_map = match([left, right, dsp_att])
        seg_concat = tf.concat([left_image, init_dsp_map, init_score_map], -1)
        dsp_concat = tf.concat([left_image, init_dsp_map, init_seg_map], -1)

        # bidirectional fusion module
        fusion = Fusion(filters=self.filters, num_class=self.num_class)
        [seg_residual, dsp_residual] = fusion([seg_concat, dsp_concat])

        score_map = init_score_map + seg_residual
        score_map = tf.math.softmax(score_map, -1)
        dsp_map = init_dsp_map + dsp_residual



        self.model = keras.Model(inputs=[left_image, right_image],
                                 outputs=[init_score_map, score_map, init_dsp_map, dsp_map])
        self.model.summary()


    def train(self, train_dir, val_dir, log_dir, weights_path, epochs, batch_size, weights):
        all_train_left_paths = glob.glob(train_dir + '/Left/*')
        all_train_right_paths = glob.glob(train_dir + '/Right/*')
        all_train_label_paths = glob.glob(train_dir + '/Cls/*')
        all_train_disp_paths = glob.glob(train_dir + '/Disparity/*')
        all_val_left_paths = glob.glob(val_dir + '/Left/*')
        all_val_right_paths = glob.glob(val_dir + '/Right/*')
        all_val_label_paths = glob.glob(val_dir + '/Cls/*')
        all_val_disp_paths = glob.glob(val_dir + '/Disparity/*')

        all_train_left_paths.sort()
        all_train_right_paths.sort()
        all_train_label_paths.sort()
        all_train_disp_paths.sort()
        all_val_left_paths.sort()
        all_val_right_paths.sort()
        all_val_label_paths.sort()
        all_val_disp_paths.sort()

        tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        lr = keras.callbacks.LearningRateScheduler(schedule=schedule, verbose=1)
        mc = keras.callbacks.ModelCheckpoint(filepath=weights_path, monitor='val_tf.__operators__.add_1_loss',
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode='min', save_freq='epoch')

        optimizer = keras.optimizers.Adam()

        unique_labels = tf.unique(all_val_label_paths)

        loss = [keras.losses.SparseCategoricalCrossentropy(),
                keras.losses.SparseCategoricalCrossentropy(),
                SmoothL1Loss(-1.0*self.max_disp, 1.0*self.max_disp),
                SmoothL1Loss(-1.0*self.max_disp, 1.0*self.max_disp)]
        loss_weights = [1.2, 1.2, 1.0, 1.0]
        if weights is not None:
            self.model.load_weights(filepath=weights, by_name=True)
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

        self.model.fit_generator(
            generator=load_batch(all_train_left_paths, all_train_right_paths, all_train_label_paths, all_train_disp_paths, batch_size, True),
            steps_per_epoch=len(all_train_left_paths)//batch_size, epochs=epochs, callbacks=[tb, lr, mc],
            validation_data=load_batch(all_val_left_paths, all_val_right_paths, all_val_label_paths, all_val_disp_paths, 1, False),
            validation_steps=len(all_val_left_paths), shuffle=False)




if __name__ == '__main__':

    sit='train'
    if sit=='train':# training
        train_dir = r'G:\3D\BGA-CFP-Net\DATA\training_fixed_jax_oma\train'
        val_dir = r'G:\3D\BGA-CFP-Net\DATA\training_fixed_jax_oma\val'
        log_dir = r'G:\3D\BGA-CFP-Net\logs'
        weights_path = r'G:\3D\BGA-CFP-Net\pre_model\PSMNet_GLFormer\mixed_attention\BGA-Net{epoch:003d}.h5'
        epochs = 80
        batch_size = 1
        # weights = r"D:\3D\HMSM-Net-master\US3D\weights\BGANet\BGANet.h5"
        weights = None
        net = BGANet(1024, 1024, 3, 96, 6, 16)
        net.build_model()
        net.train(train_dir, val_dir, log_dir, weights_path, epochs, batch_size, weights)





