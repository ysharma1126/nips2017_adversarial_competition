"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import numpy as np
from PIL import Image

import tensorflow as tf
from timeit import default_timer as timer

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 20, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


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
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')

def main(_):
  full_start = timer()

  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  num_iter = 36
  alpha = 1. * (eps / num_iter)
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  all_images_taget_class = load_target_class(FLAGS.input_dir)

  import models

  sess = []
  x_input = []
  y_input = []
  x_adv = []
    
  all_models = [models.InceptionV3Model, models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model]
  #weights = [0.2, 0.4, 0.4]
  #all_models = [models.InceptionResNetV2Model, models.InceptionV3Model, models.ResNetV2_152_Model, models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model]
 

  for i, model in enumerate(all_models):

    graph = tf.Graph()
    sess.append(tf.Session(graph=graph))
    with graph.as_default():
      # Prepare graph
      x_input.append(tf.placeholder(tf.float32, shape=batch_shape))
      x_max = tf.clip_by_value(x_input[i] + eps, -1.0, 1.0)
      x_min = tf.clip_by_value(x_input[i] - eps, -1.0, 1.0)
      y_input.append(tf.placeholder(tf.int32, shape=[FLAGS.batch_size]))

      all_models[i] = model(num_classes)
      all_models[i](x_input[i], FLAGS.batch_size)
      all_vars = slim.get_model_variables()
      model_vars = [k for k in all_vars if k.name.startswith(all_models[i].ckpt)]
      var_dict = {v.op.name[len(all_models[i].ckpt) + 1:]: v for v in model_vars}
      saver = tf.train.Saver(var_dict)

      label_mask = tf.one_hot(y_input[i], 1001, on_value=1.0, off_value=0.0, dtype=tf.float32)
      
      loss = tf.losses.softmax_cross_entropy(label_mask,all_models[i].logits,label_smoothing=0.1,weights=1.0)
      loss += tf.losses.softmax_cross_entropy(label_mask,all_models[i].aux_logits,label_smoothing=0.1,weights=0.4)
      
      #loss = tf.losses.softmax_cross_entropy(label_mask,all_models[i].logits)

      grad = tf.gradients(loss, x_input[i])[0]

      x_adv.append(tf.clip_by_value(x_input[i] - alpha * tf.sign(grad), x_min, x_max))

      saver.restore(sess[i], FLAGS.checkpoint_path + '/' + all_models[i].ckpt)

  print("Attack/Models initialized after {} sec".format(timer() - full_start))

  # Run computation
  tot_time = 0.0
  processed = 0.0
  for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    target_class_for_batch = (
        [all_images_taget_class[n] for n in filenames]
        + [0] * (FLAGS.batch_size - len(filenames)))
    input_images = images
    #print(np.array(input_images).shape)
    start = timer()
    for i in range(num_iter):
      #print('Iter',i)
      adv_images_ = []
      for j in range(len(all_models)):
        adv_images_.append(sess[j].run(x_adv[j], feed_dict={x_input[j]: input_images, y_input[j]: target_class_for_batch}))
      #adv_images_.append(adv_images_[1])
      #adv_images_.append(adv_images_[2])
      input_images = np.mean(adv_images_, axis=0)
    adv_images = input_images
    save_images(adv_images, filenames, FLAGS.output_dir)
    end = timer()
    tot_time += end - start
    processed += FLAGS.batch_size

  full_end = timer()
  print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))

if __name__ == '__main__':
  tf.app.run()
