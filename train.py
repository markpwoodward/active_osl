import argparse
import tensorflow as tf

import data
#import data_omniglot as data
import model

class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
  def __init__(self, estimator, input_fn, name):
    self.estimator = estimator
    self.input_fn = input_fn
    self.name = name

  def after_save(self, session, global_step):
    print("RUNNING EVAL: {}".format(self.name))
    self.estimator.evaluate(self.input_fn, name=self.name)
    print("FINISHED EVAL: {}".format(self.name))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--logdir", type=str, default="./logs/paper", help="The path to the directory where models and metrics should be logged.")
parser.add_argument("--batch_size", type=int, default=50, help="The number of episodes used in one gradient descent step.")
parser.add_argument("--batches_per_eval", type=int, default=20, help="The number of batches to run for each evaluation.")
parser.add_argument("--num_labels", type=int, default=3, help="The number of label slots to predict.")
parser.add_argument("--classes_per_episode", type=int, default=3, help="The number of classes in an episode.")
parser.add_argument("--time_steps", type=int, default=30, help="The number of time steps in each episode.")
parser.add_argument("--reward_correct", type=float, default=1.0, help="The reward for correctly labeling an example.")
parser.add_argument("--reward_incorrect", type=float, default=-1.0, help="The reward for incorrectly labeling an example.")
parser.add_argument("--reward_request", type=float, default=-0.05, help="The reward for requesting the label for an example.")
parser.add_argument("--epsilon_greedy", type=float, default=0.05, help="The probability the agent to randomly explore during training.")
parser.add_argument("--discount_factor", type=float, default=0.5, help="The discount factor for future rewards.")
parser.add_argument("--num_lstm_units", type=int, default=200, help="The number of units in the lstm layer.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate for training.")
params = parser.parse_args()

run_config = tf.estimator.RunConfig(
  model_dir=params.logdir,
  save_checkpoints_steps=1000,
)

estimator = tf.estimator.Estimator(model_fn=model.model_fn, config=run_config, params=params)

tf.logging.set_verbosity('INFO')

estimator.train(
  input_fn=lambda:data.input_fn(eval=False, use_validation_set=False, params=params),
  max_steps=100000,
  saving_listeners=[
    EvalCheckpointSaverListener(estimator, lambda:data.input_fn(eval=True, use_validation_set=True, params=params), "validation"),
    EvalCheckpointSaverListener(estimator, lambda:data.input_fn(eval=True, use_validation_set=False, params=params), "train"),
  ],
)
    
