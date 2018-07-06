"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import cv2 # pip install opencv-python
from scipy.ndimage.interpolation import rotate
from preprocessing.vgg_preprocessing import _random_crop, _central_crop

from scipy.stats import mode
import numpy as np
from PIL import Image
import StringIO

import tensorflow as tf

from timeit import default_timer as timer

class DatasetMetadata(object):
  """Helper class which loads and stores dataset metadata."""

  def __init__(self, filename):
    """Initializes instance of DatasetMetadata."""
    self._true_labels = {}
    self._target_classes = {}
    with open(filename) as f:
      reader = csv.reader(f)
      header_row = next(reader)
      try:
        row_idx_image_id = header_row.index('ImageId')
        row_idx_true_label = header_row.index('TrueLabel')
        row_idx_target_class = header_row.index('TargetClass')
      except ValueError:
        raise IOError('Invalid format of dataset metadata.')
      for row in reader:
        if len(row) < len(header_row):
          # skip partial or empty lines
          continue
        try:
          image_id = row[row_idx_image_id]
          self._true_labels[image_id] = int(row[row_idx_true_label])
          self._target_classes[image_id] = int(row[row_idx_target_class])
        except (IndexError, ValueError):
          raise IOError('Invalid format of dataset metadata')

  def get_true_label(self, image_id):
    """Returns true label for image with given ID."""
    return self._true_labels[image_id]

  def get_target_class(self, image_id):
    """Returns target class for image with given ID."""
    return self._target_classes[image_id]

  def save_target_classes(self, filename):
    """Saves target classed for all dataset images into given file."""
    with open(filename, 'w') as f:
      for k, v in self._target_classes.items():
        f.write('{0}.png,{1}\n'.format(k, v))


slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'csv_file', '', 'Data csv file.')

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
      #buffer = StringIO.StringIO()
      #image.save(buffer, "JPEG", quality=15)
      #image = Image.open(buffer)

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


def reduce_bit_depth(x, bit_depth):
  max_val = float(pow(2, bit_depth) - 1)
  x_int = np.rint(x * max_val)
  x_float = x_int / max_val
  return x_float


def main(_):
  start_time = timer()

  data = DatasetMetadata(FLAGS.csv_file)

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
    
    all_models = [models.InceptionResNetV2Model, models.EnsAdvInceptionResNetV2Model, models.InceptionV3Model, models.AdvInceptionV3Model, models.InceptionV4Model, models.ResNetV2_152_Model, models.ResNetV1_152_Model]
    #all_models = [models.InceptionResNetV2Model, models.EnsAdvInceptionResNetV2Model, models.AdvInceptionV3Model, models.ResNetV2_152_Model]
    #all_models = [models.InceptionResNetV2Model, models.EnsAdvInceptionResNetV2Model, models.InceptionV4Model]

    #all_models = [models.VGG16]
    #all_models = [models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model]
    #all_models = [models.InceptionResNetV2Model, models.ResNetV1_152_Model]
    #all_models = [models.InceptionResNetV2Model, models.EnsAdvInceptionResNetV2Model]

    """
    # build all the models and specify the saver
    for i, model in enumerate(all_models):
      all_models[i] = model(num_classes)
      all_models[i](x_input, FLAGS.batch_size)
      all_vars = slim.get_model_variables()
      savers.append(tf.train.Saver(set(all_vars) - initialized_vars))
      initialized_vars = set(all_vars)
    """
    for i, model in enumerate(all_models):
      all_models[i] = model(num_classes)
      all_models[i](x_input, FLAGS.batch_size)
      all_vars = slim.get_model_variables()
      model_vars = [k for k in all_vars if k.name.startswith(all_models[i].ckpt)]
      var_dict = {v.op.name[len(all_models[i].ckpt) + 1:]: v for v in model_vars}
      savers.append(tf.train.Saver(var_dict))
    # predictions from each model
    #pred = [model.preds for model in all_models]

    with tf.Session() as sess:

      for model, saver in zip(all_models, savers):
        saver.restore(sess, FLAGS.checkpoint_path + '/' + model.ckpt)
      
      pred = [model.preds for model in all_models]
      #res = []
      #res_target = []
      
      
      res = [[] for model in all_models]
      res_target = [[] for model in all_models]
      
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
       image_ids = [f.split('.')[0] for f in filenames]
       true_labels = np.array([data.get_true_label(img_id) for img_id in image_ids])
       target_labels = np.array([data.get_target_class(img_id) for img_id in image_ids])

       # Different kinds of blurring
       #images = np.array([-1.0 + 2 * cv2.GaussianBlur(0.5 * (img + 1.0), (5, 5), 0) for img in images])
       #images = np.array([cv2.medianBlur(img.astype(np.float32), 5) for img in images])
       #images = np.array([-1.0 + 2 * cv2.bilateralFilter(0.5 * (img.astype(np.float32) + 1.0), 9, 75, 75) for img in images])
       #images = np.array([np.round(127.5 * (img + 1.0) / 128.0).astype(np.int) * 128.0 / 127.5 -1.0 for img in images])
       #images = np.array([reduce_bit_depth(img, 6) for img in images])

       # Rotation
       #images = np.array([rotate(img, 20, reshape=False, mode='nearest') for img in images])
       """
       preds = sess.run(pred, feed_dict={x_input: images})
       preds = np.array(preds)
       temp = preds
       print('Shape', preds.shape)
       preds = np.concatenate((preds,np.reshape(temp[0], (1,FLAGS.batch_size))))
       preds = np.concatenate((preds,np.reshape(temp[0], (1,FLAGS.batch_size))))
       preds = np.concatenate((preds,np.reshape(temp[1], (1,FLAGS.batch_size))))
       preds = np.concatenate((preds,np.reshape(temp[1], (1,FLAGS.batch_size))))
       preds = np.concatenate((preds,np.reshape(temp[2], (1,FLAGS.batch_size))))
       preds = np.concatenate((preds,np.reshape(temp[3], (1,FLAGS.batch_size))))
       print('Shape', preds.shape)
       labels = mode(preds, axis=0)[0][0]
       labels = labels[:len(true_labels)]
       res.extend(np.array(true_labels == labels)) 
       res_target.extend(np.array(target_labels == labels)) 
       """
       
       for i, model in enumerate(all_models):
        labels = []

        #pred = [model.preds]
        pred = tf.nn.softmax(model.logits)
        preds = sess.run(pred, feed_dict={x_input: images})
        preds = np.argmax(preds, axis=1)
        preds = preds[:len(true_labels)]
        res[i].extend(np.array(true_labels == preds))
        res_target[i].extend(np.array(target_labels == preds))
      
  
  for i, model in enumerate(all_models):
    print(model.ckpt)
    print('Untargeted Score = {}'.format(100.0 - 100.0 * np.mean(res[i])))
    print('Targeted Score = {}'.format(100.0 * np.mean(res_target[i])))
  
  
  #print('Untargeted Score = {}'.format(100.0 - 100.0 * np.mean(res)))
  #print('Targeted Score = {}'.format(100.0 * np.mean(res_target)))
  
  end_time = timer()
  print("Processed all images in {} sec".format(end_time-start_time))
if __name__ == '__main__':
  tf.app.run()
