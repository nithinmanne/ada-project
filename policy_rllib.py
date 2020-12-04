from ray.rllib.policy.policy import Policy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.tf_ops import make_tf_callable
import numpy as np
import tensorflow as tf


from model import DQNRLlibModel
from config import *


def build_model(policy: Policy, obs_space, action_space, config):
    policy.model = DQNRLlibModel(obs_space, action_space, action_space.n,
                                 config['model'], name='model')
    policy.func_vars = policy.model.variables()
    policy.target_model = DQNRLlibModel(obs_space, action_space, action_space.n,
                                        config['model'], name='target_model')
    return policy.model


def build_losses(policy: Policy, _1, _2, train_batch):
    action_sample = train_batch['actions']
    state_sample = train_batch['obs']
    state_next_sample = train_batch['new_obs']
    rewards_sample = train_batch['rewards']
    done_sample = train_batch['dones']

    q_old = policy.target_model({'obs': state_next_sample})[0]
    if DQN:
        q_eval = policy.model({'obs': state_next_sample})[0]
        max_actions = tf.argmax(q_eval, axis=1)
        max_actions_oh = tf.one_hot(max_actions, policy.action_space.n)
        selected_q_old = tf.reduce_sum(q_old*max_actions_oh, 1)
    else:
        selected_q_old = tf.reduce_max(q_old, axis=1)

    q = policy.model({'obs': state_sample})[0]
    action_sample_oh = tf.one_hot(action_sample, policy.action_space.n)
    scaled_selected_q_old = tf.expand_dims(rewards_sample + (1 - tf.cast(done_sample, tf.float32))
                                           * tf.cast(GAMMA, tf.float32) * selected_q_old, axis=1)
    q_target = (1 - action_sample_oh) * q + action_sample_oh * scaled_selected_q_old
    return LOSS_FUNCTION(q_target, q)


def after_init(policy: Policy, _1, _2, _3):
    @make_tf_callable(policy.get_session())
    def update_target():
        update = []
        for v, tv in zip(policy.model.variables(), policy.target_model.variables()):
            update.append(tv.assign(v))
        return tf.group(*update)
    policy.update_target = update_target


DQNRLlibPolicy = build_tf_policy(
    name='DQNRLlibPolicy',
    get_default_config=lambda: RLLIB_DEFAULT_CONFIG,
    make_model=build_model,
    loss_fn=build_losses,
    optimizer_fn=lambda *_, **__: OPTIMIZER_V1,
    after_init=after_init
)
