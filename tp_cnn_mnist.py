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

tf.logging.set_verbosity(tf.logging.INFO)

# Using Tensorpacks ...

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
		with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):

			# following 6 layer architecture used previously
			c0 = Conv2D('conv0', image)
			p0 = MaxPooling('pool0', c0, 2)
			c1 = Conv2D('conv1', p0)
			p1 = MaxPooling('pool1', c1, 2)
			fc1 = FullyConnected('fc0', p1, 1024, nl=tf.nn.relu)
			fc1 = Dropout('dropout', fc1, 0.4)
			logits = FullyConnected('fc1', fc1, out_dim=10, nl=tf.identity)

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
	# Bactch size = hyperparam. others found (cross validation)
	train = BatchData(dataset.Mnist('train'), 128)
	test = BatchData(dataset.Mnist('test'), 256, remainder=True)

	train = PrintData(train)

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

	# lscpci or lspci | grep -i --color 'vga\|3d\|2d' to get graphic card ids to pass in
	# $ lspci
	# 00:00.0 Host bridge: Intel Corporation 440FX - 82441FX PMC [Natoma] (rev 02)
	# 00:01.0 ISA bridge: Intel Corporation 82371SB PIIX3 ISA [Natoma/Triton II]
	# 00:01.1 IDE interface: Intel Corporation 82371SB PIIX3 IDE [Natoma/Triton II]
	# 00:01.3 Bridge: Intel Corporation 82371AB/EB/MB PIIX4 ACPI (rev 01)
	# 00:02.0 VGA compatible controller: Cirrus Logic GD 5446
	# 00:03.0 Ethernet controller: Device 1d0f:ec20
	# 00:1d.0 VGA compatible controller: NVIDIA Corporation GM204GL [Tesla M60] (rev a1)
	# 00:1e.0 VGA compatible controller: NVIDIA Corporation GM204GL [Tesla M60] (rev a1)
	# 00:1f.0 Unassigned class [ff80]: XenSource, Inc. Xen Platform Device (rev 01)

	# thus, use the following command to run on given remote machine:
	# python tp_cnn_mnist.py --gpu 00:d.0, 00:1e.0

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
		# ref. http://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/train/trainers.html#SyncMultiGPUTrainerReplicated
		launch_train_with_config(config, SyncMultiGPUTrainer())
	else:
		# trainer info ref. http://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/train/trainers.html#SimpleTrainer
		launch_train_with_config(config, QueueInputTrainer())
