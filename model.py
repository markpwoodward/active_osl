import tensorflow as tf

import metrics

def model_fn(features, labels, mode, params):
  """Builds episode graph, the train operation, and the metric operations

  Args:
    features: (batch_size, time_steps, features...), in the case of omniglot (batch_Size, time_steps, 28, 28, 1) tf.float32
    labels: (batch_size, time_steps, num_labels) tf.int32
    mode: tf.estimator.ModeKeys.[TRAIN, EVAL, PREDICT]
    params: a Dictionary of configuration parameters

  Returns:
    tf.estimator.EstimatorSpec
  """
  features_ts = tf.unstack(features, axis=1) # (batch_size, ...)[time_steps], tf.float32
  label_ts = tf.unstack(labels, axis=1) # (batch_size, num_labels)[time_steps], tf.int32

  explore = (mode == tf.estimator.ModeKeys.TRAIN)
  agent = Agent(explore, params)
  sim = Simulator(features_ts, label_ts, params)

  a_ts, q_ts = episode(sim, agent, params)

  loss_t = tf.reduce_mean(td_error(a_ts, q_ts, label_ts, params)) # tf.float32
  
  train_op = tf.train.AdamOptimizer(params.learning_rate).minimize(loss_t, global_step=tf.train.get_global_step())

  eval_op_d = {}
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_op_d.update(metrics.accuracies(label_ts, a_ts))
    eval_op_d.update(metrics.requests(label_ts, a_ts))

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss_t, train_op=train_op, eval_metric_ops=eval_op_d)
      
def episode(sim, agent, params):
  """Builds an episode graph using a Simulator and an Agent

  Args:
    sim: a Simulator object
    agent: an Agent object
    params: application level constants

  Returns:
    (a_ts, q_ts)
    a_ts: (batch_size, num_actions)[time_steps], tf.int32, the chosen actions
    q_ts: (batch_size, num_actions)[time_steps], tf.float32, expected future return for each action
  """
  a_ts = []
  q_ts = []
  
  for t in range(params.time_steps):
    o_ts = sim.get_observation()
    a_t, q_t = agent.next_action(o_ts)

    a_ts.append(a_t)
    q_ts.append(q_t)

    sim.do_step(a_t)

  return (a_ts, q_ts)

class Simulator(object):
  def __init__(self, features_ts, label_ts, params):
    self.features_ts = features_ts
    self.label_ts = label_ts

    self.t = 0
    self.zero_label_t = tf.zeros((params.batch_size, params.num_labels), dtype=tf.int32) # (batch_size, num_labels), tf.int32
    self.last_label_t = self.zero_label_t

  def do_step(self, a_t):
    """Not much simulator state here. Just sets last_label if a label request was made"""
    request_t = tf.expand_dims(a_t[:,-1], axis=-1) # (50,1), tf.int32
    label_t = self.label_ts[self.t] # (50, num_labels), tf.int32
    self.last_label_t = request_t*label_t + (1-request_t)*self.zero_label_t # (50, num_labels), tf.int32
    self.t += 1

  def get_observation(self):
    return (self.last_label_t, self.features_ts[self.t])

class Agent(object):
  def __init__(self, explore, params):
    self.explore = explore
    self.batch_size = params.batch_size
    self.num_actions = params.num_labels+1
    self.epsilon_greedy = params.epsilon_greedy
    self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(params.num_lstm_units)
    self.rnn_initial_state_t = self.rnn_cell.zero_state(params.batch_size, tf.float32) # feed this when deploying the model
    self.rnn_state_t = self.rnn_initial_state_t
    self.reuse = False

  def next_action(self, o_ts):
    last_label_t, features_t = o_ts

    # concat last label to flattened features
    net_t = tf.concat([tf.cast(last_label_t, dtype=tf.float32), tf.layers.flatten(features_t)], axis=-1)

    # advance the rnn
    net_t, self.rnn_state_t = self.rnn_cell(net_t, self.rnn_state_t)
    
    # compute q values
    q_t = tf.layers.dense(net_t, self.num_actions, reuse=self.reuse)
      
    # choose action
    a_max_t = tf.argmax(q_t, axis=1, output_type=tf.int32) # (batch_size), tf.int32 in [0,num_actions)
    a_rand_t = tf.random_uniform([self.batch_size], maxval=self.num_actions, dtype=tf.int32) # (batch_size), tf.int32 in [0, num_actions)
    
    if self.explore:
      explore_t = tf.cast(tf.random_uniform([self.batch_size])+self.epsilon_greedy, tf.int32) # int(rand()+epsilon), p(1)=epsilon, p(0)=1-epsilon
      a_t = explore_t*a_rand_t + (1-explore_t)*a_max_t # (batch_size), tf.int32
    else:
      a_t = a_max_t # (batch_size), tf.int32

    a_t = tf.one_hot(a_t, self.num_actions, dtype=tf.int32) # (batch_size, num_actions), tf.int32

    self.reuse = True

    return (a_t, q_t)

def td_error(a_ts, q_ts, label_ts, params):
  """Computes the temporal difference error for an episode.

  Args:
    a_ts: (batch_size, num_actions)[time_steps], tf.int32, the chosen actions
    q_ts: (batch_size, num_actions)[time_steps], tf.float32, expected future return for each action
    label_ts: (batch_size, num_labels)[time_steps], tf.int32, the correct label for the example
    params: application level constants

  Returns:
    (batch_size), tf.float32, sum of td errors for all steps in the episode
  """
  a_ts = [tf.cast(a_t, tf.float32) for a_t in a_ts]
  td_error_t = tf.zeros([params.batch_size]) # (batch_size), tf.float32
  for t in range(params.time_steps-1):
    gamma = params.discount_factor
    q_t = tf.reduce_sum(q_ts[t]*a_ts[t], axis=1) # (batch_size), tf.float32
    r_t = reward(a_ts[t], label_ts[t], params) # (batch_size), tf.float32
    max_next_q_t = tf.reduce_max(q_ts[t+1], axis=1) # (batch_size), tf.float32

    td_error_t += (q_t - (r_t + gamma*max_next_q_t))**2 # (batch_size), tf.float32
    
  return td_error_t # (batch_size), tf.float32

def reward(a_t, label_t, params):
  """Extracts the reward, from params, across the batch for a single time step
  
  Args:
    a_t: (batch_size, num_actions), tf.int32, the chosen action
    label_t: (batch_size, num_labels), tf.int32, the correct label for the example
    params: application level constants

  Returns:
    (batch_size), tf.float32, the reward
  """
  a_t = tf.cast(a_t, tf.float32)
  label_t = tf.cast(label_t, tf.float32)
  rewards_t = label_t*params.reward_correct + (1-label_t)*params.reward_incorrect # (batch_size, num_labels), tf.float32
  rewards_t = tf.pad(rewards_t, [[0,0],[0,1]], constant_values=params.reward_request) # (batch_size, num_labels+1), tf.float32

  r_t = tf.reduce_sum(rewards_t*a_t, axis=1) # (batch_size), tf.float32

  return r_t

