import tensorflow as tf
from tensorflow.python.keras.layers import  MaxPooling2D, Conv2D, \
     BatchNormalization,  UpSampling2D, concatenate, Lambda,ZeroPadding2D
from tensorflow.python.keras.models import Model


class VGGNetModel(Model):
    def __init__(self):
        super(VGGNetModel, self).__init__()
        # block 1
        self.block1_conv1 = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())

        self.block1_conv2 = Conv2D(64, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block1_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
        self.block1_batch_norm = BatchNormalization(name='block1_batch_norm')

        # block 2
        self.block2_conv1 = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block2_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block2_conv2 = Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block2_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
        self.block2_batch_norm = BatchNormalization(name='block2_batch_norm')

        # Block 3
        self.block3_conv1 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block3_conv2 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block3_conv3 = Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block3_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
        self.block3_batch_norm = BatchNormalization(name='block3_batch_norm')

        # Block 4
        self.block4_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block4_conv2 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block4_conv3 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block4_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
        self.block4_batch_norm = BatchNormalization(name='block4_batch_norm')

        # Block 5
        self.blcok5_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv1',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block5_conv2 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block5_conv3 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block5_conv3',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())

        self.block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
        self.block5_batch_norm = BatchNormalization(name='block5_batch_norm')

        # Block 6
        self.block6_conv1 = Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='block6_conv1',
                                   dilation_rate=6,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block6_conv2 = Conv2D(512, (1, 1),
                                   activation='relu',
                                   padding='same',
                                   name='block6_conv2',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                   kernel_initializer=tf.keras.initializers.glorot_normal())

        self.block6_batch_norm = BatchNormalization(name='block6_batch_norm')

        # block 7
        self.block7_up_conv1 = Conv2D(512, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block7_up_conv1',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block7_batch_norm1 = BatchNormalization(name='block7_batch_norm1')
        self.block7_up_conv2 = Conv2D(256, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block7_up_conv2',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block7_batch_norm2 = BatchNormalization(name='block7_batch_norm2')
        self.block7_up_sampling = UpSampling2D(name='block7_up_sampling')
        # block 8
        self.block8_up_conv1 = Conv2D(256, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block8_up_conv1',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block8_batch_norm1 = BatchNormalization(name='block8_batch_norm1')
        self.block8_up_conv2 = Conv2D(128, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block8_up_conv2',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block8_batch_norm2 = BatchNormalization(name='block8_batch_norm2')
        self.block8_up_sampling = UpSampling2D(name='block8_up_sampling')

        # block 9
        self.block9_up_conv1 = Conv2D(128, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block9_up_conv1',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block9_batch_norm1 = BatchNormalization(name='block9_batch_norm1')
        self.block9_up_conv2 = Conv2D(64, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      name='block9_up_conv2',
                                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                      kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block9_batch_norm2 = BatchNormalization(name='block9_batch_norm2')
        self.block9_up_sampling = UpSampling2D(name='block9_up_sampling')

        # block 10
        self.block10_up_conv1 = Conv2D(64, (3, 3),
                                       activation='relu',
                                       padding='same',
                                       name='block10_up_conv1',
                                       kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                       kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block10_batch_norm1 = BatchNormalization(name='block10_batch_norm1')
        self.block10_up_conv2 = Conv2D(32, (3, 3),
                                       activation='relu',
                                       padding='same',
                                       name='block10_up_conv2',
                                       kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                       kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block10_batch_norm2 = BatchNormalization(name='block10_batch_norm2')
        self.block10_up_sampling = UpSampling2D(name='block10_up_sampling')

        # block 11
        self.block11_conv1 = Conv2D(32, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    name='block11_conv1',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block11_conv2 = Conv2D(32, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    name='block11_conv2',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block11_conv3 = Conv2D(16, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    name='block11_conv3',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block11_conv4 = Conv2D(16, (1, 1),
                                    activation='relu',
                                    padding='same',
                                    name='block11_conv4',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())
        self.block11_conv5 = Conv2D(2, (1, 1),
                                    activation='sigmoid',
                                    padding='same',
                                    name='block11_conv5',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal())

        self.padding = ZeroPadding2D(padding=((1, 0), (0, 0)), data_format='channels_last')

    def call(self, inputs, training=None, mask=None, preference=False):
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)
        x = self.block1_batch_norm(x)
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)
        x_2 = self.block2_batch_norm(x)
        print(x_2)
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pool(x)
        x_3 = self.block3_batch_norm(x)
        print(x_3)
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)
        x_4 = self.block4_batch_norm(x)
        print(x_4)
        x = self.blcok5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)
        x_5 = self.block5_batch_norm(x)
        print(x_5)
        x = self.block6_conv1(x)
        x = self.block6_conv2(x)
        x = self.block6_batch_norm(x)
        x = concatenate([x_5, x])
        x = self.block7_up_conv1(x)
        x = self.block7_batch_norm1(x)
        x = self.block7_up_conv2(x)
        x = self.block7_batch_norm2(x)
        x = self.block7_up_sampling(x)
        if x_4.shape[1] == 67:
            x = self.padding(x)

        x = concatenate([x_4, x])
        x = self.block8_up_conv1(x)
        x = self.block8_batch_norm1(x)
        x = self.block8_up_conv2(x)
        x = self.block8_batch_norm2(x)
        x = self.block8_up_sampling(x)
        if x_3.shape[1] == 135:
            x = self.padding(x)
        x = concatenate([x_3, x])
        x = self.block9_up_conv1(x)
        x = self.block9_batch_norm1(x)
        x = self.block9_up_conv2(x)
        x = self.block9_batch_norm2(x)
        x = self.block9_up_sampling(x)

        x = concatenate([x_2, x])
        x = self.block10_up_conv1(x)
        x = self.block10_batch_norm1(x)
        x = self.block10_up_conv2(x)
        x = self.block10_batch_norm2(x)
        x = self.block10_up_sampling(x)

        x = self.block11_conv1(x)
        x = self.block11_conv2(x)
        x = self.block11_conv3(x)
        x = self.block11_conv4(x)
        x = self.block11_conv5(x)

        region_score = Lambda(lambda layer: layer[..., 0])(x)
        affinity_score = Lambda(lambda layer: layer[..., 1])(x)
        return region_score, affinity_score



if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf

    model = VGGNetModel()
    model.build(input_shape=(1,768,768,3))
    #data = np.random.random((1, 768, 768, 3)).astype(np.float32)

    print(model.summary())



