import tensorflow as tf
from tensorflow import Tensor


class Inputs(object):
    def __init__(self, img1: Tensor, img2: Tensor, label: Tensor):
        self.img1 = img1
        self.img2 = img2
        self.label = label

    def feed_input(self, input_img1, input_img2, input_label=None):
        # feed the input images that are necessary to make a prediction
        feed_dict = {self.img1: input_img1,
                     self.img2: input_img2}

        # optionally also include the label:
        # if we're just making a prediction without calculating loss, that won't be necessary
        if input_label is not None:
            feed_dict[self.label] = input_label

        return feed_dict


class Model(object):
    def __init__(self, inputs: Inputs):
        self.inputs = inputs
        self.predictions = self.predict(inputs)
        self.loss = self.calculate_loss(inputs, self.predictions)
        self.opt_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self, inputs: Inputs):
        with tf.name_scope("image_substraction"):
            img_diff = (inputs.img1 - inputs.img2)
            x = img_diff
        with tf.name_scope('conv_relu_maxpool'):
            for conv_layer_i in range(5):
                x = tf.layers.conv2d(x,
                                     filters=20 * (conv_layer_i + 1),
                                     kernel_size=3,
                                     activation=tf.nn.relu)
                x = tf.layers.max_pooling2d(x,
                                            pool_size=3,
                                            strides=2)
        with tf.name_scope('fully_connected'):
            for conv_layer_i in range(1):
                x = tf.layers.dense(x,
                                    units=200,
                                    activation=tf.nn.relu)
        with tf.name_scope('linear_predict'):
            predicted_logits = tf.layers.dense(x, 1, activation=None)

        return tf.squeeze(predicted_logits)

    def calculate_loss(self, inputs: Inputs, prediction_logits: Tensor):
        with tf.name_scope('calculate_loss'):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs.label,
                                                                          logits=prediction_logits))
