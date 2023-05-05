# uncompyle6 version 3.8.0
# Python bytecode 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
# [GCC 9.4.0]
# Embedded file name: /home/azon/Documents/GitHub/AlphaZero/OOG-AlphaZero-v5/DN.py
# Compiled at: 2022-09-14 16:10:45
# Size of source mod 2**32: 6707 bytes
from tensorflow.keras.layers import Activation, Flatten, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Layer, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
import tensorflow as tf
DN_FILTERS = 64
DN_RESIDUA_NUM = 16
DN_INPUT_SHAPE = (15, 15, 2)
DN_OUTPUT_SIZE = 225

class GlobalPoolingBlock(Layer):

    def __init__(self, DN_FILTERS):
        super(GlobalPoolingBlock, self).__init__()
        self.conv = Conv2D(DN_FILTERS, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.conv1 = Conv2D((int(DN_FILTERS / 4)), kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.conv2 = Conv2D((DN_FILTERS - int(DN_FILTERS / 4)), kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.conv3 = Conv2D(DN_FILTERS, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.bn = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.relu = Activation('relu')
        self.relu1 = Activation('relu')
        self.relu2 = Activation('relu')
        self.GAP = GlobalAveragePooling2D()
        self.GMP = GlobalMaxPooling2D()
        self.fully_connect1 = Dense(DN_FILTERS - int(DN_FILTERS / 4))
        
    def call(self, input_tensor, training=False):
        x = input_tensor
        identity1 = x #input2
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        identity2 = self.conv2(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.GAP(x)
        x2 = self.GMP(x)
        x = Concatenate()([x1, x2])
        x = self.fully_connect1(x)

        x = Add()([x, identity2])

        x = self.bn2(x)
        x = self.relu2(x)
        #x = Dropout(0.3) #訓練的時候把隨機把0.3的節點拿掉
        x = self.conv3(x)
        x = Add()([x, identity1])
        return x


class ResBlock(Layer):

    def __init__(self, DN_FILTERS):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(DN_FILTERS, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(DN_FILTERS, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')

    def call(self, input_tensor, training=False):
        x = input_tensor
        identity = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = Add()([x, identity])
        return x


class Policyhead(Layer):

    def __init__(self, C_head):
        super(Policyhead, self).__init__()
        self.conv = Conv2D(C_head, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.conv1 = Conv2D(C_head, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.conv2 = Conv2D(2, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.bn = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.relu = Activation('relu')
        self.relu1 = Activation('relu')
        self.relu2 = Activation('relu')
        self.averagepool = GlobalAveragePooling2D()
        self.maximumpool = GlobalMaxPooling2D()
        self.fully_connect = Dense(C_head)

    def call(self, input_tensor, training=False):
        x = input_tensor

        identity = x
        identity = self.bn1(identity)
        identity = self.relu1(identity)
        identity = self.conv1(identity)

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x1 = self.averagepool(x)
        x2 = self.maximumpool(x)
        x = Concatenate()([x1, x2])
        x = self.fully_connect(x)

        x = Add()([x, identity])

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class Valuehead(Layer):

    def __init__(self, C_head, C_value):
        super(Valuehead, self).__init__()
        self.conv2 = Conv2D(C_head, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)))
        self.relu2 = Activation('relu')
        self.averagepool = GlobalAveragePooling2D()
        self.maximumpool = GlobalMaxPooling2D()
        self.fully_connect1 = Dense(C_value, kernel_regularizer=(L2(0.0005)), activation='relu', name='v')
        self.fully_connect2 = Dense(1, kernel_regularizer=(L2(0.0005)), activation='tanh', name='v')

    def call(self, input_tensor, training=False):
        x = self.conv2(input_tensor)
        x = self.relu2(x)
        x = self.averagepool(x)
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x


class DN(Model):

    def __init__(self):
        super(DN, self).__init__()
        self.resblock1 = ResBlock(64)
        self.resblock2 = ResBlock(64)
        self.resblock3 = ResBlock(64)
        self.resblock4 = ResBlock(64)
        self.resblock5 = ResBlock(64)
        self.resblock6 = ResBlock(64)
        self.resblock7 = ResBlock(64)
        self.globalpoolingblock1 = GlobalPoolingBlock(64)
        self.globalpoolingblock2 = GlobalPoolingBlock(64)
        self.globalpoolingblock3 = GlobalPoolingBlock(64)
        self.policy_head = Policyhead(18)
        self.value_head = Valuehead(18, 27)
        self.flatten1 = Flatten(name='my_policy')
        self.flatten2 = Flatten(name='emeny_policy')
        self.conv = Conv2D(DN_FILTERS, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=(L2(0.0005)), name='input')
        self.bn = BatchNormalization(name='BN')
        self.relu = Activation('relu', name='relu')
        #self.my_policy = Activation('softmax', name='my_policy')
        #self.emeny_policy = Activation('softmax', name='emeny_policy')

    @tf.function()
    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.globalpoolingblock1(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.globalpoolingblock2(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.globalpoolingblock3(x)
        
        x = self.bn(x)
        x = self.relu(x)

        v = self.value_head(x)
        p = self.policy_head(x)

        p1 = self.flatten1(p[:, :, :, 0])
        p2 = self.flatten2(p[:, :, :, 1])
        #p1 = self.my_policy(p1)
        #p2 = self.emeny_policy(p2)

        return [p1, p2, v]

    def model(self):
        x = Input(shape = DN_INPUT_SHAPE)
        return Model(inputs=[x], outputs=(self.call(x)))