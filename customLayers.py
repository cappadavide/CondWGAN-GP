from keras import layers
import tensorflow as tf

class UpsampleBlock(layers.Layer):
    def __init__(self,filters, activation, kernel_size=(3,3),strides=(1,1),up_size=(2,2),padding="same",use_bn=False, use_bias=True, use_dropout=False, drop_value=0.3,**kwargs):
        super(UpsampleBlock, self).__init__(name='upsampleblock')
        self.upsampling = layers.UpSampling2D(up_size)
        self.conv2d = layers.Conv2D(filters,kernel_size,strides=strides, padding=padding, use_bias=use_bias)
        self.activation = activation
        self.bn = None  
        self.dropout = None
        if use_bn:
            self.bn = layers.BatchNormalization() 
        if use_dropout:
            self.dropout = layers.Dropout(drop_value)

        super(UpsampleBlock, self).__init__(**kwargs)


    def call(self, inputs):
        x = self.upsampling(inputs)
        x = self.conv2d(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'upsampling': self.upsampling,
            'conv2d': self.conv2d,
            'activation': self.activation,
            'bn': self.bn,
            'dropout': self.dropout,

        })
        return config


class ConvBlock(layers.Layer):
    def __init__(self,filters, activation, kernel_size=(3,3),strides=(1,1),padding="same",use_bn=False, use_bias=True, use_dropout=False, drop_value=0.3,**kwargs):
        super(ConvBlock, self).__init__(name='convblock')
        self.conv2d = layers.Conv2D(filters,kernel_size,strides=strides, padding=padding, use_bias=use_bias)
        self.activation = activation
        self.bn = None  
        self.dropout = None
        if use_bn:
            self.bn = layers.BatchNormalization() 
        if use_dropout:
            self.dropout = layers.Dropout(drop_value)
        super(ConvBlock, self).__init__(**kwargs)

    def call(self, inputs):
        x = self.conv2d(inputs)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv2d': self.conv2d,
            'activation': self.activation,
            'bn': self.bn,
            'dropout': self.dropout,

        })
        return config