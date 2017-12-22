import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn_graph_model as dqn
from dqn_utils import *
from atari_wrappers import *
from knapsack_env import *
# import Q_function_graph_model
import Q_function_graph_model2 as Q_function_graph_model
import mvc_env


def graph_learn(env, num_timesteps, q_func):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return False

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=q_func,
        pre_pooling_mlp_layers=2,
        post_pooling_mlp_layers=1,
        n_hidden_units=64, T=4,
        initialization_stddev=1e-3,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=128,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        target_update_freq=2000,
        grad_norm_clipping=10,
        double_DQN=True,
        n_steps_ahead=3,
        learning_rate=1e-4
    )
    env.close()


def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[2]

    # Run training
    env = mvc_env.MVC_env(7)

    graph_learn(env, num_timesteps=task.max_timesteps,
                q_func=Q_function_graph_model.Q_func)

if __name__ == "__main__":
    main()
