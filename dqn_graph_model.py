import sys
import gym.spaces
import itertools
import numpy as np
import random
import logz
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import replay_buffer_graph
import time

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

"""
learn slightly modified to pass the task name as an argument
so that it is easier to record data.
"""

def learn(env,
          q_func,
          pre_pooling_mlp_layers,
          post_pooling_mlp_layers,
          n_hidden_units,
          T=4,
          initialization_stddev=1e-4,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          double_DQN=True,
          n_steps_ahead=3,
          learning_rate=1e-3
         ):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    exp_name = env.env_name
    logz.configure_output_dir('data/' + exp_name + time.strftime('%m-%d-%Y-%H:%M:%s'))
    ###############
    # BUILD MODEL #
    ###############

    input_shape = env.state_shape
    num_actions = env.num_actions

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.float32, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32, [None], name='act_t_ph')
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.float32, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])
    transition_length_ph = tf.placeholder(tf.float32, [None])

    # Graphs specific placeholder
    adj_ph = tf.placeholder(tf.float32, [None, env.number_nodes, env.number_nodes],
                            name='adj_ph')
    graph_weights_ph = tf.placeholder(tf.float32,
                                      [None, env.number_nodes, env.number_nodes],
                                      name='graph_weights_ph')

    q_func_net = q_func(x=obs_t_ph,
                        adj=adj_ph,
                        w=graph_weights_ph,
                        p=n_hidden_units, T=T, initialization_stddev=initialization_stddev,
                        scope="q_func", reuse=False,
                        pre_pooling_mlp_layers=pre_pooling_mlp_layers,
                        post_pooling_mlp_layers=post_pooling_mlp_layers)
    q_func_net_argmax_target = q_func(x=obs_tp1_ph,
                                      adj=adj_ph,
                                      w=graph_weights_ph,
                                      p=n_hidden_units, T=T, initialization_stddev=initialization_stddev,
                                      scope="q_func", reuse=False,
                                      pre_pooling_mlp_layers=pre_pooling_mlp_layers,
                                      post_pooling_mlp_layers=post_pooling_mlp_layers)
    target_q_func_net = q_func(x=obs_tp1_ph,
                               adj=adj_ph,
                               w=graph_weights_ph,
                               p=n_hidden_units, T=T, initialization_stddev=initialization_stddev,
                               scope="target_q_func", reuse=False,
                               pre_pooling_mlp_layers=pre_pooling_mlp_layers,
                               post_pooling_mlp_layers=post_pooling_mlp_layers)

    if not double_DQN:
        target_y = rew_t_ph + tf.pow(gamma, transition_length_ph) *\
                              done_mask_ph * tf.reduce_max(target_q_func_net, axis=1)
    else:
        target_y = rew_t_ph + \
                   tf.pow(gamma, transition_length_ph) * done_mask_ph * \
                   tf.reduce_sum(target_q_func_net *\
                                 tf.one_hot(tf.argmax(q_func_net_argmax_target, axis = 1),
                                            depth=num_actions),\
                                 axis=1)

    actual_y = tf.reduce_sum(tf.multiply(q_func_net, tf.one_hot(act_t_ph, depth=num_actions)), axis=1)
    total_error = tf.nn.l2_loss(target_y - actual_y)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='target_q_func')

    training_error_summ_sy = tf.summary.scalar('training_total_error', total_error)

    # construct optimization op (with gradient clipping)
    # learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_fn = optimizer.minimize(total_error)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = replay_buffer_graph.ReplayBuffer(replay_buffer_size, obs_size=input_shape[0],
                                                     n_nodes=input_shape[0])

    # Model saver
    saver = tf.train.Saver()

    # Create session, initialize variables
    session = tf.InteractiveSession()
    log_files_name = 'DQN_' + str(env.env_name) + \
                     '-lf=' + str(learning_freq) + \
                     '-b=' + str(batch_size) + '-' + \
                     time.strftime('%m-%d-%Y-%H:%M:%S')

    writer = tf.summary.FileWriter('/tmp/' + log_files_name,
                                   session.graph)
    tf.global_variables_initializer().run()
    saver.save(session, '/tmp/saved_models/' + log_files_name)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    observations = [env.reset()]
    LOG_EVERY_N_STEPS = 10000

    episode_total_rewards = []
    episode_total_optimal_rewards = []
    episode_total_at_random_rewards = []
    accuracies = []
    done = False
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        import random
        from numpy import array

        if done:
            observations = [env.reset()]

        # Choose action
        if model_initialized:
            epsilon = exploration.value(t)
            q_values=session.run(q_func_net, feed_dict={obs_t_ph: observations[-1][None],
                                                        adj_ph: env.adjacency_matrix[None],
                                                        graph_weights_ph: env.weight_matrix[None]})

            action = np.argmax(q_values[0] * (1 - observations[-1]) - 1e5 * observations[-1])
            r = random.random()
            if r <= epsilon:
                all_possible_action = list(range(num_actions))
                # other_actions = [x for x in all_possible_action if x != action]
                if env.env_name == 'MVC':
                    other_actions = [x for x in all_possible_action if env.state[x] != 1]
                else:
                    other_actions = [x for x in all_possible_action if observations[-1][x] != 1]
                action = np.array(random.choice(other_actions))
        else:
            action = np.array(random.choice(list(range(num_actions))))

        next_obs, reward, done = env.step(action)
        observations.append(next_obs)

        if len(observations) > n_steps_ahead:
            replay_buffer.store_transition(observations[-(n_steps_ahead + 1)], env.adjacency_matrix,
                                           env.weight_matrix, action, reward, observations[-1], done, n_steps_ahead)

        if done and 1 < len(observations) <= n_steps_ahead:
            replay_buffer.store_transition(observations[0], env.adjacency_matrix, env.weight_matrix,
                                           action, reward, observations[-1], done, len(observations) - 1)


        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken

        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            obs_t_batch, adj_batch, graph_weights_batch, act_batch,\
            rew_batch, obs_tp1_batch, done_mask_batch, transition_length_batch = replay_buffer.sample(batch_size)
            if not(model_initialized):
                initialize_interdependent_variables(session, tf.global_variables(), {
                            obs_t_ph: obs_t_batch,
                            obs_tp1_ph: obs_tp1_batch,
                        })
                model_initialized=True

            training_error_summ, _ = session.run([training_error_summ_sy, train_fn],
                                                 feed_dict={obs_t_ph: obs_t_batch,
                                                            adj_ph: adj_batch,
                                                            graph_weights_ph: graph_weights_batch,
                                                            act_t_ph: act_batch,
                                                            rew_t_ph:rew_batch,
                                                            obs_tp1_ph: obs_tp1_batch,
                                                            done_mask_ph: done_mask_batch,
                                                            transition_length_ph: transition_length_batch})

            # import pdb; pdb.set_trace()
            if t % 100:
                writer.add_summary(training_error_summ, t)
                writer.flush()

            if num_param_updates%target_update_freq == 0:
                session.run(update_target_fn)
            num_param_updates += 1

            #####

            ### 4. Log progress
            # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if done:
                episode_total_rewards.append(env.accumulated_reward())
                episode_total_optimal_rewards.append(env.optimal_solution()[0])
                episode_total_at_random_rewards.append(env.at_random_solution()[0])


            if len(episode_total_rewards) > 0:
                mean_episode_reward = np.mean(np.array(episode_total_rewards)[-1000:])
                mean_optimal_episode_reward = np.mean(np.array(episode_total_optimal_rewards)[-1000:])
                mean_at_random_episode_reward = np.mean(np.array(episode_total_at_random_rewards)[-1000:])
                if env.env_name == 'TSP':
                    mean_approx_ratio = np.mean(np.array(episode_total_rewards)[-1000:] /
                                                np.mean(np.array(episode_total_optimal_rewards)[-1000:]))

            if len(episode_total_rewards) > 1000:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
                # Save the model
                saver.save(session, '/tmp/saved_models/' + log_files_name, global_step=t)
                # Display and log episode stats
                logz.log_tabular("Timestep", t)
                logz.log_tabular("AtRandomAverageReturn", mean_at_random_episode_reward)
                logz.log_tabular("AverageReturn", mean_episode_reward)
                logz.log_tabular("OptimalAverageReturn", mean_optimal_episode_reward)
                if env.env_name == 'TSP':
                    logz.log_tabular("ApproxRatio", mean_approx_ratio)
                logz.log_tabular("MaxReturn", best_mean_episode_reward)
                logz.log_tabular("Episodes", len(episode_total_rewards))
                logz.log_tabular("Exploration", exploration.value(t))
                logz.dump_tabular()

                sys.stdout.flush()