"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
from scipy.stats import mode

import tensorflow as tf

from PIL import Image
import StringIO
import cv2

slim = tf.contrib.slim


tf.flags.DEFINE_string(
		'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
		'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
		'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
		'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
		'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
		'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
		'batch_size', 20, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape):
	"""Read png images from input directory in batches.

	Args:
		input_dir: input directory
		batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

	Yields:
		filenames: list file names without path of each image
			Lenght of this list could be less than batch_size, in this case only
			first few images of the result are elements of the minibatch.
		images: array with all images from this batch
	"""
	images = np.zeros(batch_shape)
	filenames = []
	idx = 0
	batch_size = batch_shape[0]
	for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
		with tf.gfile.Open(filepath) as f:
			image = Image.open(f)
			# JPEG COMPRESSION
			buffer = StringIO.StringIO()
			image.save(buffer, "JPEG", quality=25)
			image = Image.open(buffer)

		# Images for inception classifier are normalized to be in [-1, 1] interval.
		images[idx, :, :, :] = np.array(image.convert('RGB')).astype(np.float) / 255.0 * 2.0 - 1.0

		filenames.append(os.path.basename(filepath))
		idx += 1
		if idx == batch_size:
			yield filenames, images
			filenames = []
			images = np.zeros(batch_shape)
			idx = 0
	if idx > 0:
		yield filenames, images





def main(_):
	
	batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
	num_classes = 1001

	tf.logging.set_verbosity(tf.logging.INFO)

	with tf.Graph().as_default():
		# Prepare graph
		x_input = tf.placeholder(tf.float32, shape=batch_shape)

		import models

		initialized_vars = set()
		savers = []

		# list of models in our ensemble
		all_models = [models.InceptionResNetV2Model, models.EnsAdvInceptionResNetV2Model, models.AdvInceptionV3Model, models.ResNetV2_152_Model]

		# build all the models and specify the saver
		for i, model in enumerate(all_models):
			all_models[i] = model(num_classes)
			all_models[i](x_input, FLAGS.batch_size)
			all_vars = slim.get_model_variables()
			model_vars = [k for k in all_vars if k.name.startswith(all_models[i].ckpt)]
			var_dict = {v.op.name[len(all_models[i].ckpt) + 1:]: v for v in model_vars}
			savers.append(tf.train.Saver(var_dict))

		with tf.Session() as sess:
			with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
				for model, saver in zip(all_models, savers):
					saver.restore(sess, FLAGS.checkpoint_path + '/' + model.ckpt)

				pred = [model.preds for model in all_models]
				res = []
				res_target = []

				for filenames, images in load_images(FLAGS.input_dir, batch_shape):
				 image_ids = [f.split('.')[0] for f in filenames]

				 # Different kinds of blurring
				 #images = np.array([-1.0 + 2 * cv2.GaussianBlur(0.5 * (img + 1.0), (5, 5), 0) for img in images])
				 #images = np.array([cv2.medianBlur(img.astype(np.float32), 5) for img in images])
				 #images = np.array([-1.0 + 2 * cv2.bilateralFilter(0.5 * (img.astype(np.float32) + 1.0), 9, 75, 75) for img in images])
				 #images = np.array([np.round(127.5 * (img + 1.0) / 128.0).astype(np.int) * 128.0 / 127.5 -1.0 for img in images])
				 #images = np.array([reduce_bit_depth(img, 6) for img in images])

				 # Rotation
				 #images = np.array([rotate(img, 20, reshape=False, mode='nearest') for img in images])

				 preds = sess.run(pred, feed_dict={x_input: images})
				 preds = np.array(preds)
				 temp = preds
				 print('Shape', preds.shape)

				 preds = np.concatenate((preds,np.reshape(temp[0], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[0], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[0], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[0], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[0], (1,FLAGS.batch_size))))

				 preds = np.concatenate((preds,np.reshape(temp[1], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[1], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[1], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[1], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[1], (1,FLAGS.batch_size))))

				 preds = np.concatenate((preds,np.reshape(temp[2], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[2], (1,FLAGS.batch_size))))

				 preds = np.concatenate((preds,np.reshape(temp[3], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[3], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[3], (1,FLAGS.batch_size))))
				 preds = np.concatenate((preds,np.reshape(temp[3], (1,FLAGS.batch_size))))
				 
				 print('Shape', preds.shape)
				 labels = mode(preds, axis=0)[0][0]
				 """
				 labels = labels[:len(true_labels)]
				 res.extend(np.array(true_labels == labels)) 
				 res_target.extend(np.array(target_labels == labels)) 
				 """
				 for filename, label in zip(filenames, labels):
					out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
	tf.app.run()
