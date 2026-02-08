import tensorflow.keras as keras
import tensorflow as tf


class Conv_block(tf.keras.Model):

    def __init__(self,  filter):
        super(Conv_block, self).__init__()

        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, inputs):
        x = self.conv(inputs)
        return x
class Conv_block3(tf.keras.Model):

    def __init__(self, filter):
        super(Conv_block3, self).__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x):
        x = self.conv(x)
        return x

class SEBlock(keras.Model):
    def __init__(self, filter, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.Sequential([
            keras.layers.Dense(filter // reduction, activation='relu'),
            keras.layers.Dense(filter, activation='sigmoid')
        ])

    def call(self, inputs):
        x = inputs
        y = self.avg_pool(x)
        y = tf.expand_dims( tf.expand_dims(y, axis=1), axis=1)  # match shape of x
        y = self.fc(y)
        return x * y


class GlobalAttention_1(keras.Model):
    def __init__(self):
        super(GlobalAttention_1, self).__init__()
        self.gamma = tf.Variable(initial_value=tf.zeros((1,)), trainable=True,name='gl_gamma_1')
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x):

        batch_size, height, width, channels= x.shape
        proj_query = tf.reshape(x, (1, channels, height*width))
        proj_key = tf.transpose(tf.reshape(x, (1, channels, -1)), perm=(0, 2, 1))
        energy = tf.matmul(proj_query, proj_key)

        energy_max = tf.reduce_max(energy, axis=-1, keepdims=True)
        energy_new = energy_max - energy
        attention = self.softmax(energy_new)

        proj_value = tf.reshape(x, (1, channels, -1))

        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, (1,  height, width,channels))


        out = self.gamma * out + x
        return out

class GlobalAttention_2(keras.Model):
    def __init__(self):
        super(GlobalAttention_2, self).__init__()
        self.gamma = tf.Variable(initial_value=tf.zeros((1,)), trainable=True,name='gl_gamma_2')
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x):

        batch_size, height, width, channels= x.shape
        proj_query = tf.reshape(x, (1, channels, height*width))
        proj_key = tf.transpose(tf.reshape(x, (1, channels, -1)), perm=(0, 2, 1))
        energy = tf.matmul(proj_query, proj_key)

        energy_max = tf.reduce_max(energy, axis=-1, keepdims=True)
        energy_new = energy_max - energy
        attention = self.softmax(energy_new)

        proj_value = tf.reshape(x, (1, channels, -1))

        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, (1,  height, width,channels))


        out = self.gamma * out + x
        return out

class GlobalAttention_3(keras.Model):
    def __init__(self):
        super(GlobalAttention_3, self).__init__()
        self.gamma = tf.Variable(initial_value=tf.zeros((1,)), trainable=True,name='gl_gamma_3')
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x):

        batch_size, height, width, channels= x.shape
        proj_query = tf.reshape(x, (1, channels, height*width))
        proj_key = tf.transpose(tf.reshape(x, (1, channels, -1)), perm=(0, 2, 1))
        energy = tf.matmul(proj_query, proj_key)

        energy_max = tf.reduce_max(energy, axis=-1, keepdims=True)
        energy_new = energy_max - energy
        attention = self.softmax(energy_new)

        proj_value = tf.reshape(x, (1, channels, -1))

        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, (1,  height, width,channels))


        out = self.gamma * out + x
        return out

class GlobalAttention_4(keras.Model):
    def __init__(self):
        super(GlobalAttention_4, self).__init__()
        self.gamma = tf.Variable(initial_value=tf.zeros((1,)), trainable=True,name='gl_gamma_4')
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x):

        batch_size, height, width, channels= x.shape
        proj_query = tf.reshape(x, (1, channels, height*width))
        proj_key = tf.transpose(tf.reshape(x, (1, channels, -1)), perm=(0, 2, 1))
        energy = tf.matmul(proj_query, proj_key)

        energy_max = tf.reduce_max(energy, axis=-1, keepdims=True)
        energy_new = energy_max - energy
        attention = self.softmax(energy_new)

        proj_value = tf.reshape(x, (1, channels, -1))

        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, (1,  height, width,channels))


        out = self.gamma * out + x
        return out    






class LocalAttention(tf.keras.Model):
    def __init__(self, filter):
        super(LocalAttention, self).__init__()
        self.conv1 = Conv_block(filter)
        self.conv2 = Conv_block3(filter)
        self.se = SEBlock(filter)

    def call(self, inputs):
        x_cov1 = self.conv1(inputs)
        x_cov2 = self.conv2(inputs)
        x = self.se(x_cov1 + x_cov2)
        return x


class GlobalLocalAttention_1(tf.keras.Model):
    def __init__(self, filter):
        super(GlobalLocalAttention_1, self).__init__()

        self.GA = GlobalAttention_1()
        self.LA = LocalAttention(filter)
        self.cov = Conv_block3(filter)

    def call(self, x):
        x_g = self.GA(x)
        x_l = self.LA(x)
        print(x_g.shape)
        print(x_l.shape)
        x = self.cov(x_g + x_l)

        return x


class GlobalLocalAttention_2(tf.keras.Model):
    def __init__(self, filter):
        super(GlobalLocalAttention_2, self).__init__()

        self.GA = GlobalAttention_2()
        self.LA = LocalAttention(filter)
        self.cov = Conv_block3(filter)

    def call(self, x):
        x_g = self.GA(x)
        x_l = self.LA(x)
        x = self.cov(x_g + x_l)

        return x

class GlobalLocalAttention_3(tf.keras.Model):
    def __init__(self, filter):
        super(GlobalLocalAttention_3, self).__init__()

        self.GA = GlobalAttention_3()
        self.LA = LocalAttention(filter)
        self.cov = Conv_block3(filter)

    def call(self, x):
        x_g = self.GA(x)
        x_l = self.LA(x)
        x = self.cov(x_g + x_l)

        return x

class GlobalLocalAttention_4(tf.keras.Model):
    def __init__(self, filter):
        super(GlobalLocalAttention_4, self).__init__()

        self.GA = GlobalAttention_4()
        self.LA = LocalAttention(filter)
        self.cov = Conv_block3(filter)

    def call(self, x):
        x_g = self.GA(x)
        x_l = self.LA(x)
        x = self.cov(x_g + x_l)

        return x

class DecoderBlock(tf.keras.Model):
    def __init__(self, in_channels, n_filters, nonlinearity=tf.nn.relu):
        super(DecoderBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(in_channels // 4, (1, 1), padding='same')
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = nonlinearity

        self.deconv2 = tf.keras.layers.Conv2DTranspose(in_channels // 4, (3, 3), strides=(2, 2), padding='same', output_padding=(1, 1))
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = nonlinearity

        self.conv3 = tf.keras.layers.Conv2D(n_filters, (1, 1), padding='same')
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = nonlinearity

    def call(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

layers = tf.keras.layers


class conv_block_resnet(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, strides=(2, 2)):
        super(conv_block_resnet, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # 降维
        self.conv1 = layers.Conv2D(filters1, (1, 1), strides=strides,
                          name=conv_name_base + 'my_2a')
        self.bn1 = layers.BatchNormalization(name=bn_name_base + 'my_2a')
        self.act1 = layers.Activation('relu')

        # 3x3卷积
        self.conv2 = layers.Conv2D(filters2, kernel_size, padding='same',
                          name=conv_name_base + 'my_2b')
        self.bn2 = layers.BatchNormalization(name=bn_name_base + 'my_2b')
        self.act2 = layers.Activation('relu')

        # 升维
        self.conv3 = layers.Conv2D(filters3, (1, 1), name=conv_name_base + 'my_2c')
        self.bn3 = layers.BatchNormalization(name=bn_name_base + 'my_2c')

        # 残差边
        self.shortcut1 = layers.Conv2D(filters3, (1, 1), strides=strides,
                                 name=conv_name_base + 'my_1')
        self.shortcut2  = layers.BatchNormalization(name=bn_name_base + 'my_1')


        self.act3 = layers.Activation('relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut1(inputs)
        shortcut = self.shortcut2(shortcut)
        x = layers.add([x, shortcut])
        x = self.act3(x)

        return x




class identity_block(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block):
        super(identity_block, self).__init__(name='')

        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # 降维
        self.conv1 = layers.Conv2D(filters1, (1, 1), name=conv_name_base + 'my_2a')
        self.bn1 = layers.BatchNormalization(name=bn_name_base + 'my_2a')
        self.act1 = layers.Activation('relu')
        # 3x3卷积
        self.conv2 = layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + 'my_2b')
        self.bn2 = layers.BatchNormalization(name=bn_name_base + 'my_2b')
        self.act2 = layers.Activation('relu')
        # 升维
        self.conv3 = layers.Conv2D(filters3, (1, 1), name=conv_name_base + 'my_2c')
        self.bn3 = layers.BatchNormalization(name=bn_name_base + 'my_2c')


        self.act3 = layers.Activation('relu')

    def call(self, inputs):
            x=self.conv1(inputs)
            x=self.bn1(x)
            x=self.act1(x)
            x=self.conv2(x)
            x=self.bn2(x)
            x=self.act2(x)
            x=self.conv3(x)
            x = self.bn3(x)

            # shortcut=self.shortcut1(inputs)
            # shortcut = self.shortcut2(shortcut)
            x = layers.add([x, inputs])
            x=self.act3(x)

            return x


class ResNet50(tf.keras.Model):
    def __init__(self):
        super(ResNet50, self).__init__(name='')

        self.conv = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',name='my_conv1')
        self.bn = layers.BatchNormalization(name='my_bn_conv1')
        self.act = layers.Activation('relu')

        # [56,56,64]
        self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same')

        # [56,56,256]
        self.conv_block1 = conv_block_resnet( 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.identity_block1 = identity_block( 3, [64, 64, 256], stage=2, block='b')
        self.identity_block2 = identity_block( 3, [64, 64, 256], stage=2, block='c')

        # [28,28,512]
        self.conv_block2 = conv_block_resnet( 3, [128, 128, 512], stage=3, block='a')
        self.identity_block3 = identity_block( 3, [128, 128, 512], stage=3, block='b')
        self.identity_block4 = identity_block( 3, [128, 128, 512], stage=3, block='c')
        self.identity_block5 = identity_block( 3, [128, 128, 512], stage=3, block='d')

        # [14,14,1024]
        self.conv_block3 = conv_block_resnet( 3, [256, 256, 1024], stage=4, block='a')
        self.identity_block6 = identity_block( 3, [256, 256, 1024], stage=4, block='b')
        self.identity_block7 = identity_block( 3, [256, 256, 1024], stage=4, block='c')
        self.identity_block8 = identity_block( 3, [256, 256, 1024], stage=4, block='d')
        self.identity_block9 = identity_block( 3, [256, 256, 1024], stage=4, block='e')
        self.identity_block10 = identity_block( 3, [256, 256, 1024], stage=4, block='f')

        # [7,7,2048]
        self.conv_block4 = conv_block_resnet(3, [512, 512, 2048], stage=5, block='a')
        self.identity_block11 = identity_block( 3, [512, 512, 2048], stage=5, block='b')
        self.identity_block12=identity_block(3, [512, 512, 2048], stage=5, block='c')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x1 = self.conv_block1(x)
        x1 = self.identity_block1(x1)
        x1 = self.identity_block2(x1)

        x2 = self.conv_block2(x1)
        x2 = self.identity_block3(x2)
        x2 = self.identity_block4(x2)
        x2 = self.identity_block5(x2)

        x3 = self.conv_block3(x2)
        x3 = self.identity_block6(x3)
        x3 = self.identity_block7(x3)
        x3 = self.identity_block8(x3)
        x3 = self.identity_block9(x3)
        x3 = self.identity_block10(x3)

        x4 = self.conv_block4(x3)
        x4 = self.identity_block11(x4)
        x4 = self.identity_block12(x4)

        return x1, x2, x3, x4