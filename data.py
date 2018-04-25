"""
The omniglot dataset consists 20 hand drawn examples from each of 1623 characters 
from alphabets around the world. The original image size is 105x105.

The omniglot dataset is preprocessed to 28x28 and split into 1200 training characters
and 423 validation characters. omniglot.npz is a compressed (zipfile) archive containing
train.npy and validate.npy.
* train.npy is a np.uint8 array of size (1200,20,28,28)
* validate.npy is a np.uint8 array of size (423,20,28,28)
"""
import random
import numpy as np
import tensorflow as tf

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
DATA_FILE_PATH = "./omniglot.npz"

def input_fn(eval, use_validation_set, params):
  """Outputs a tuple containing a features tensor and a labels tensor

  Args:
    eval: whether we are evaluating or training. Training generates episodes indefinitely, evaluating generates params.batches_per_evaluation.
    use_validation_set: bool, if True then load the validation set, otherwise load the training set.
    params: a dictionary of configuration parameters

  Returns:
    (features_t, labels_t)
  """
  ds = tf.data.Dataset.from_generator(
    lambda:episode_batch_generator(use_validation_set, params),
    (tf.float32, tf.int32),
    ((params.batch_size, params.time_steps, IMAGE_HEIGHT, IMAGE_WIDTH), (params.batch_size, params.time_steps, params.num_labels)),
  )
  if eval:
    ds = ds.take(params.batches_per_eval)
  ds = ds.prefetch(4) # 4 is arbitrary, a little prefetching helps speed things up

  images_t, labels_t = ds.make_one_shot_iterator().get_next()

  return (images_t, labels_t)

def episode_batch_generator(use_validation_set, params):
  """Yields a batch of episode images and their corresponding labels

  Args:
    use_validation_set: bool, whether to build the episode from the train or validation images
    params: application level constants

  Yields:
    A tuple (images, labels)
    * images: (batch_size, time_steps, image_height, image_width), np.float32
    * labels: (batch_size, time_steps, num_labels), np.int32
  """
  while True:
    yield get_episode_batch(use_validation_set, params)

def get_episode_batch(use_validation_set, params):
  """Batches up params.batch_size episodes

  Args:
    use_validation_set: bool, whether to build the episode from the train or validation images
    params: application level constants

  Returns:
    A tuple (images, labels)
    * images: (batch_size, time_steps, image_height, image_width), np.float32
    * labels: (batch_size, time_steps, num_labels), np.int32
  """
  images, labels = zip(*[get_episode(use_validation_set, params) for _ in range(params.batch_size)])
  return np.array(images), np.array(labels)

_data = {} # cache so that we don't keep reloading
def get_episode(use_validation_set, params):
  """Creates an episode

  Creates an episode by randomly choosing params.classes_per_episode classes, then drawing
  params.time_steps images from those classes, then randomly ordering the resulting images.

  Each class is randomnly assigned (without replacement) a label from params.num_labels.

  Args:
    use_validation_set: bool, whether to build the episode from the train or validation images
    params: application level constants

  Returns:
    A tuple (images, labels)
    * images: (time_steps, image_height, image_width), np.float32
    * labels: (time_steps, num_labels), np.int32 "one-hot"
  """
  if len(_data) == 0:
    _data.update(np.load(DATA_FILE_PATH))
    
  dataset_images = _data["validate"] if use_validation_set else _data["train"]
  num_classes, examples_per_class, _, _ = dataset_images.shape

  # choose classes
  classes = random.sample(range(num_classes), params.classes_per_episode)
  
  # choose labels
  class_labels = random.sample(range(params.classes_per_episode), params.classes_per_episode)
  
  # choose rotation for each class
  class_rotation = np.random.choice(range(4), params.classes_per_episode)

  # choose images
  # NOTE: this is slower than it could be, too much sampling I think
  samples_per_class = random.sample(list(range(params.classes_per_episode))*examples_per_class, params.time_steps) # e.g. [0,1,1,0,2,0,0,1]
  indices = [random.sample(range(examples_per_class), samples_per_class.count(i)) for i in range(params.classes_per_episode)] # e.g. [[18,5,10,3], [17,9,12], [19]]
  labels = [[class_labels[c]]*len(cs) for c, cs in enumerate(indices)] # e.g. [ [2,2,2,2], [0,0,0], [1] ]
  labels = [item for sublist in labels for item in sublist] # e.g. [2,2,2,2,0,0,0,1]

  indices = [zip([classes[c]]*len(cs), cs) for c, cs in enumerate(indices)] # e.g. [ [[128,18], [128,5], [128,10], [128,3]], [[55,17],[55,9],[55,12]], [[91,19]] ]
  indices = [item for sublist in indices for item in sublist] # e.g. [ [128,18], [128,5], [128,10], [128,3], [55,17], [55,9], [55,12], [91,19] ]

  shuffled_order = random.sample(range(params.time_steps), params.time_steps) # e.g. [7, 3, 2, 4, 5, 1, 6, 0]
  labels = [labels[i] for i in shuffled_order] # e.g. [1, 2, 2, 0, 0, 2, 0, 2]
  indices = [indices[i] for i in shuffled_order] # e.g. [ [91,19], [128,3], [128,10], [55,17], [55,9], [128,5], [55,12], [128,18] ]

  indices = list(zip(*indices)) # e.g. [ [91, 128, 128, 55, 55, 128, 55, 128], [19, 3, 10, 17, 9, 5, 12, 18] ]
  images_raw = dataset_images[indices[0], indices[1], :, :] # (time_steps, raw_image_height, raw_image_width)

  # augment images
  images = np.zeros([params.time_steps, IMAGE_HEIGHT, IMAGE_WIDTH], dtype=np.float32)
  for i in range(params.time_steps):
    im = images_raw[i]
    
    # class rotation (0, pi/2, pi, 3*pi/2)
    im = np.rot90(im, k=class_rotation[labels[i]])

    # # mild rotation (-pi/16, pi/16), TOO SLOW, maybe do this in tensorflow
    # im = scipy.misc.imrotate(im, np.random.random()*(np.pi/8.0)-(np.pi/16.0))
    
    # # translate (+/- 10 pixels), TOO SLOW, maybe do this in tensorflow
    # im = np.pad(im, 10, 'constant', constant_values=0.0)
    # offset = np.random.randint(20, size=2)
    # im = im[offset[0]:offset[0]+IMAGE_HEIGHT, offset[1]:offset[1]+IMAGE_WIDTH]

    images[i] = im

  # scale images to [0,1]
  images = images/255.0

  # insert extra labels that are never used. Fills classes_per_episode up to num_labels.
  if params.classes_per_episode < params.num_labels:
    mapping = random.sample(range(params.num_labels), params.classes_per_episode)
    labels = [mapping[label] for label in labels]

  # convert labels to one-hot
  labels = np.eye(params.num_labels, dtype=np.int32)[labels] # (num_labels) np.int32

  return (images, labels)

if __name__ == "__main__":
  import argparse
  import PIL
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=50, help="The number of episodes used in one gradient descent step.")
  parser.add_argument("--batches_per_eval", type=int, default=20, help="The number of batches to run for each evaluation.")
  parser.add_argument("--num_labels", type=int, default=3, help="The number of label slots to predict.")
  parser.add_argument("--classes_per_episode", type=int, default=3, help="The number of classes in an episode.")
  parser.add_argument("--time_steps", type=int, default=30, help="The number of time steps in each episode.")
  params = parser.parse_args()

  # verify shape of episode generator output
  images, labels = get_episode(use_validation_set=False, params=params)
  assert images.shape == (params.time_steps, 28, 28), "incorrect episode generator output, images"
  assert labels.shape == (params.time_steps, params.num_labels), "incorrect episode generator output shape, labels"
  print("images.shape: {} = {}".format(images.shape, (params.time_steps, 105, 105)))
  print("labels.shape: {} = {}".format(labels.shape, (params.time_steps, params.num_labels)))

  # verify that evaluation only runs params.batches_per_eval batches
  images_t, labels_t = input_fn(eval=True, use_validation_set=False, params=params)
  n_batches = 0
  with tf.Session() as sess:
    while True:
      try:
        sess.run([images_t, labels_t])
      except:
        break
      n_batches += 1
  assert n_batches == params.batches_per_eval
  print("n_batches: {} = {}".format(n_batches, params.batches_per_eval))

  print("All tests passed")

  raise SystemExit # comment this if you want to continue and view frames from an episode
  
  # view one random episode from a batch
  images_t, labels_t = input_fn(eval=False, use_validation_set=True, params=params)
  with tf.Session() as sess:
    images, labels = sess.run([images_t, labels_t])
    print(images.shape)
    print(labels.shape)

    batch_size, time_steps, height, width = images.shape
    i_b = random.randrange(batch_size)
    
    for t in range(time_steps):
      im = images[i_b, t, :, :]
      im = (im*255.0).astype(np.uint8)
      PIL.Image.fromarray(im).show()
      input("label = {} > ".format(labels[i_b, t, :]))
