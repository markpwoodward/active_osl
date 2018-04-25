import numpy as np
import tensorflow as tf

def accuracies(label_ts, a_ts):
  """precision_t, accuracy_t, accuracy_1st_t, accuracy_2nd_t, accuracy_5th_t, accuracy_10th_t"""

  def accuracies_np(labels, actions):
    batch_size, time_steps, num_labels = labels.shape
    
    labels = np.argmax(labels, axis=-1) # (batch_size, time_steps) np.int32
    actions = np.argmax(actions, axis=-1) # (batch_size, time_steps) np.float32

    # precision
    precision = np.sum(labels == actions)/np.sum(actions != num_labels) if np.sum(actions != num_labels) > 0 else np.array(0.0)

    # accuracy
    accuracy = np.mean(labels == actions)

    episode_counts = np.zeros([batch_size, time_steps], np.int) # the instance number for each time step
    count = np.zeros([batch_size, num_labels], np.int) # a running count of each label in the episode
    for t in range(time_steps):
      # increment
      count[range(batch_size),labels[:,t]] += 1
      # label the step
      episode_counts[:, t] = count[range(batch_size),labels[:,t]]
    
    # 1st
    ids = episode_counts == 1
    accuracy_1st = np.mean(labels[ids] == actions[ids])
  
    # 2nd
    ids = episode_counts == 2
    accuracy_2nd = np.mean(labels[ids] == actions[ids])
  
    # 5th
    ids = episode_counts == 5
    accuracy_5th = np.mean(labels[ids] == actions[ids])

    # 10th
    ids = episode_counts == 10
    accuracy_10th = np.mean(labels[ids] == actions[ids])

    ret = [arr.astype(np.float32) for arr in [precision, accuracy, accuracy_1st, accuracy_2nd, accuracy_5th, accuracy_10th]]

    return ret
    
  accuracies_ts = tf.py_func(accuracies_np, [tf.stack(label_ts, axis=1), tf.stack(a_ts, axis=1)], [tf.float32]*6)
  accuracies_d = {
    "precision": tf.metrics.mean(accuracies_ts[0]),
    "accuracy": tf.metrics.mean(accuracies_ts[1]),
    "accuracy_01st": tf.metrics.mean(accuracies_ts[2]),
    "accuracy_02nd": tf.metrics.mean(accuracies_ts[3]),
    "accuracy_05th": tf.metrics.mean(accuracies_ts[4]),
    "accuracy_10th": tf.metrics.mean(accuracies_ts[5]),
  }

  return accuracies_d
  
def requests(label_ts, a_ts):
  """requests_t, requests_1st_t, requests_2nd_t, requests_5th_t, requests_10th_t"""

  def requests_np(labels, actions):
    batch_size, time_steps, num_labels = labels.shape

    labels = np.argmax(labels, axis=-1) # (batch_size, time_steps) np.int32
    actions = np.argmax(actions, axis=-1) # (batch_size, time_steps) np.float32

    # requests
    requests = np.mean(actions == num_labels)

    episode_counts = np.zeros([batch_size, time_steps], np.int)
    count = np.zeros([batch_size, num_labels], np.int)
    for t in range(time_steps):
      # increment
      count[range(batch_size),labels[:,t]] += 1
      # label the step
      episode_counts[:, t] = count[range(batch_size),labels[:,t]]
    
    # 1st
    ids = episode_counts == 1
    requests_1st = np.mean(actions[ids] == num_labels)
  
    # 2nd
    ids = episode_counts == 2
    requests_2nd = np.mean(actions[ids] == num_labels)
    
    # 5th
    ids = episode_counts == 5
    requests_5th = np.mean(actions[ids] == num_labels)

    # 10th
    ids = episode_counts == 10
    requests_10th = np.mean(actions[ids] == num_labels)
    
    ret = [arr.astype(np.float32) for arr in [requests, requests_1st, requests_2nd, requests_5th, requests_10th]]

    return ret

  requests_ts = tf.py_func(requests_np, [tf.stack(label_ts, axis=1), tf.stack(a_ts, axis=1)], [tf.float32]*5)
  requests_d = {
    "requests": tf.metrics.mean(requests_ts[0]),
    "requests_01st": tf.metrics.mean(requests_ts[1]),
    "requests_02nd": tf.metrics.mean(requests_ts[2]),
    "requests_05th": tf.metrics.mean(requests_ts[3]),
    "requests_10th": tf.metrics.mean(requests_ts[4]),
  }

  return requests_d

