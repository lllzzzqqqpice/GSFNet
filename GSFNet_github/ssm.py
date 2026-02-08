import tensorflow.keras as keras
from ssm_submodule import *
import tensorflow as tf
from fem import *
from tensorflow.keras import layers, models

class GLFormer(tf.keras.Model):
    def __init__(self, filters, num_classes=5):
        super(GLFormer, self).__init__()

        self.trans1 = FeatureTrans(8 * filters)
        self.trans2 = FeatureTrans_down(16 * filters)
        self.trans3 = FeatureTrans_down(32 * filters)
        self.trans4 = FeatureTrans_down(64 * filters)

        self.attention4 = GlobalLocalAttention_1(64*filters)
        self.attention3 = GlobalLocalAttention_2(32*filters)
        self.attention2 = GlobalLocalAttention_3(16*filters)
        self.attention1 = GlobalLocalAttention_4(8*filters)

        self.decoder4 = DecoderBlock(64*filters, 32*filters)
        self.decoder3 = DecoderBlock(32*filters, 16*filters)
        self.decoder2 = DecoderBlock(16*filters, 8*filters)
        self.decoder1 = DecoderBlock(8*filters, 8*filters)

        self.conv_add = layers.Conv2D(8*filters, kernel_size=3,strides=2, padding='same')
        self.deconv_add = layers.Conv2DTranspose(8*filters, kernel_size=3, strides=2, padding='same')

        self.finaldeconv1 = layers.Conv2DTranspose(2*filters, kernel_size=4, strides=2, padding='same')
        self.finalrelu1 = tf.nn.relu
        self.finalconv2 = layers.Conv2D(2*filters, kernel_size=3, padding='same')
        self.finalrelu2 = tf.nn.relu
        self.finalconv3 = layers.Conv2D(num_classes, kernel_size=3, padding='same')


    def call(self, inputs):

        e1 = self.trans1(inputs[0])

        e2 = self.trans2(e1)

        e3 = self.trans3(e2)

        e4 = self.trans4(e3)


        e4 = self.attention4(e4)
        # Decoder
        d4 = self.decoder4(e4) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)

        d1=self.conv_add(d1)

        d1= d1*inputs[1]+d1

        d1 = self.deconv_add(d1)

        out = self.finaldeconv1(d1)

        out = self.finalrelu1(out)
        out = self.finalconv2(out)

        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


