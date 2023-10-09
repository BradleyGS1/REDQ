import tensorflow as tf
from keras.layers import Layer
from keras import initializers


class EnsembleDenseLayer(Layer):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(EnsembleDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        trainable=True)
        
        super(EnsembleDenseLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        x, zero_mask = inputs
        output = tf.matmul(x * zero_mask, self.kernel)
        if self.use_bias:
            output = tf.add(output, self.bias * zero_mask)
        return self.activation(output), zero_mask

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer
        })
        
        return config
    
class EnsembleSplitLayer(Layer):
    def __init__(self, **kwargs):
        super(EnsembleSplitLayer, self).__init__(**kwargs)

    def build(self, input_shape):        
        super(EnsembleSplitLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.split(inputs, inputs.shape[-1], axis=-1)

class EnsembleOutputLayer(Layer):
    def __init__(self, **kwargs):
        super(EnsembleOutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):        
        super(EnsembleOutputLayer, self).build(input_shape)

    def call(self, inputs):
        x, zero_mask = inputs
        return x + (1 - zero_mask) * 1e8