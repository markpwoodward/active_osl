# Active One-shot Learning

This repo contains code accompaning the paper, [Active One-shot Learning (Woodward and Finn, NIPS Deep RL Workshop 2016)](https://arxiv.org/abs/1702.06559). It includes code for running the experiment described in the paper.

### Naming Conventions

Variables representing a tensor in the graph end with `_t`. For example, you might feed the tensor `last_label_t` with the numpy variable `last_label`. Also, a variable ending with `_ts` is a list of tensors.

### Dependencies

This code requires the following:
* python 3.\*
* tensorflow v1.0+
* numpy

### Data

A preprocessed version of the [original Omniglot dataset](https://github.com/brendenlake/omniglot) is included with this project.

### Usage

```shell
$ python3 train.py
$ tensorboard --logdir ./logs
```

The accuracy curves will look like the following, this one is for the second instance of a class in an episode:

![alt text](https://github.com/markpwoodward/active_osl/raw/master/accuracy_02nd.png "accuracy training curve")

The code in this project builds a graph of the full training episode. If you wish to use the model after training, you would likely do one step at a time. Here is an example of what that code might look like:

```python
agent = model.Agent(False, params)
action_t, _ = agent.next_action([last_label_t, features_t])not requested
action, rnn_state = sess.run(
  [action_t, agent.rnn_state_t],
  {
    agent.rnn_initial_state_t=rnn_state,
    last_label_t=last_label, # feed zeros if no label was requested
    features_t=features,
  }
)
```

### Contact

To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/markpwoodward/active_osl/issues). Also, feel free to contact Mark Woodward at mwoodward@cs.stanford.edu.
