import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class GradCam:
    def __init__(self, model, last_conv_layer_name):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name

    def __call__(self, img_array):
        with tf.GradientTape() as tape:
            last_conv_layer = self.model.get_layer(self.last_conv_layer_name)
            iterate = tf.keras.models.Model([self.model.inputs], [self.model.output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(img_array)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap
