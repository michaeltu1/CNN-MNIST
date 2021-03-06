from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

import os
import argparse
from tensorpack import *
from tensorpack.tfutils import summary, get_current_tower_context
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu

#import horovod.tensorflow as hvd

tf.logging.set_verbosity(tf.logging.INFO)

# Using Tensorpacks ...
batch_size = 128
num_gpus = 1
NR_GPU = 2

# ref. https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/graph_builder/model_desc.py for inheritance info
class Model(ModelDesc):

    def inputs(self):
        """
        __Create__ and returns a list of placeholders.
        A subclass is expected to implement this method.
        The placeholders __have to__ be created inside this method.
        Don't return placeholders created in other methods.
        Also, you should not call this method by yourself.
        Returns:
            a list of `tf.placeholder`, to be converted to :class:`InputDesc`.
        """
        # what inputs do I need? images
        # what should be defined as image_size? 
        # ref. http://tensorpack.readthedocs.io/en/latest/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.Mnist
        # images are 28x28
        return [tf.placeholder(tf.float32, (None, 28, 28), 'input'), tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, image, label):
        """
        Build the whole symbolic graph.
        This is supposed to be part of the "tower function" when used with :class:`TowerTrainer`.
        By default it will call :meth:`_build_graph` with a list of input tensors.
        A subclass is expected to overwrite this method or the :meth:`_build_graph` method.
        Args:
            args ([tf.Tensor]): tensors that matches the list of inputs defined by ``inputs()``.
        Returns:
            In general it returns nothing, but a subclass (e.g.
            :class:`ModelDesc`) may require it to return necessary information
            (e.g. cost) to build the trainer.
        """
        # inputs to conv nets are NWHC := Num_samples x Height x Width x Channels
        image = tf.expand_dims(image, 3)

        image = image * 2 - 1   # center the pixels values at zero?? i don't understand ..

        # build symbolic layers somewhere in here
        # ref. info about argscope: http://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/tfutils/argscope.html
        # making layers in argscope is supposed to let you do something ..? assign arg. characteristics to each layer
        # tp layers
        """
        #with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):

        # following 6 layer architecture used previously
        c0 = Conv2D('conv0', image, kernel_size=3, nl=tf.nn.relu, out_channel=32)
        # c0.variables = None
        p0 = MaxPooling('pool0', c0, 2)
        # p0.variables = None
        c1 = Conv2D('conv1', p0, kernel_size=3, nl=tf.nn.relu, out_channel=32)
        # c1.variables = None
        p1 = MaxPooling('pool1', c1, 2)
        # p1.variables = None
        fc1 = FullyConnected('fc0', p1, 1024, nl=tf.nn.relu)
        # fc1.variables = None
        fc1 = Dropout('dropout', fc1, rate=0.6)
        # fc1.variables = None
        logits = FullyConnected('fc1', fc1, out_dim=10, nl=tf.identity)
        # logits.variables = None
        """
        # tf layers
        conv1 = tf.layers.conv2d(
        inputs=image,
        filters=32,
        kernel_size=3,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        padding="same",
        activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=3,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        padding="same",
        activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)
        #"""

        # Should I have this line if I'm doing sparse_softmax_cross_entropy_with_logits later?
        tf.nn.softmax(logits, name='prob') # normalize to usable prob. distr.

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        # Casts to float32 type after checking if the prediction (1st) is equal to the label value
        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensosrboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error') # ?
        summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # Regularizing - avoiding overfitting
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms'])) # ?
        return total_cost

    def optimizer(self):
        # Returns a `tf.train.Optimizer` instance.
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    # Dataflow / Input src
    # Batch size = hyperparam. others found (cross validation)
    train = BatchData(dataset.Mnist('train'), batch_size / num_gpus)
    test = BatchData(dataset.Mnist('test'), 2 * batch_size / num_gpus, remainder=True)

    # train = PrintData(train)

    print("Testing Dataflow Speed ...")
    print(TestDataSpeed(dataset.Mnist('train')).start())
    print("Ended Dataflow test")

    return train, test


def get_config():
    dataset_train, dataset_test = get_data()
    # How many iterations you want in each epoch.
    # This is the default value, don't actually need to set it in the config
    steps_per_epoch = dataset_train.size()

    # get the config which contains everything necessary in a training
    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,  # the DataFlow instance for training
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # automatically setup the directory train_log/tp_cnn_mnist for logging
    logger.auto_set_dir()

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)

    if args.gpu:
        print("Using MultiGPUTrainer ...")
        nr_tower = get_nr_gpu()
        launch_train_with_config(config, SyncMultiGPUTrainer(nr_tower))
    else:
        print("Using QueueInputTrainer ...")
        launch_train_with_config(config, QueueInputTrainer())

