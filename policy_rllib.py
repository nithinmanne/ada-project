"""This contains code for creating the RLlib policy for DQN. The logic is mostly
   similar to the agent, but because of the RLlib works, it needs to implemented
   in a specific manner."""
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.tf_ops import make_tf_callable


from model import DQNRLlibModel
from config import *


def build_model(policy: Policy, obs_space, action_space, config):
    """This creates both the prediction and target models and attaches
       them to the policy"""
    policy.model = DQNRLlibModel(obs_space, action_space, action_space.n,
                                 config['model'], name='model')
    policy.func_vars = policy.model.variables()
    policy.target_model = DQNRLlibModel(obs_space, action_space, action_space.n,
                                        config['model'], name='target_model')
    return policy.model


def build_losses(policy: Policy, _1, _2, train_batch):
    """This function creates the graph that will be used to generate
       the losses on the prediction network. This function is executed
       with placeholder inputs so all the code should comply with that.
       The actual logic is same as the training algorithm in agent.py"""
    action_sample = train_batch['actions']
    state_sample = train_batch['obs']
    state_next_sample = train_batch['new_obs']
    rewards_sample = train_batch['rewards']
    done_sample = train_batch['dones']

    q_old = policy.target_model({'obs': state_next_sample})[0]
    if DDQN:
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
    """This function is called after the above two finished executing, it sets up
       the code to update the target network. Like the other codes, there are no actual
       values while running this and it needs to be written in such a way that the graph
       has the object to write the weights, than actually writing here. In fact, unlike the
       previous function, there aren't even any placeholder inputs, which means that this
       function doesn't even know anything about the graph created and writing code normally
       would just update the weights this one time, and not do anything else when RLlib simply
       tries to execute a graph function that does nothing. This is why the decorator provided
       by RLlib is used to let TensorFlow know that any operations called in this code are not
       actual operations, but rather to add these operations to the graph that's part of the policy."""
    @make_tf_callable(policy.get_session())
    def update_target():
        update = []
        for v, tv in zip(policy.model.variables(), policy.target_model.variables()):
            update.append(tv.assign(v))
        return tf.group(*update)
    """The RLlib API expects the name of this function to be update_target, and is something
       I found out only by reading the actual source code."""
    policy.update_target = update_target


"""The plicy created using all the above definitions."""
DQNRLlibPolicy = build_tf_policy(
    name='DQNRLlibPolicy',
    get_default_config=lambda: RLLIB_DEFAULT_CONFIG,
    make_model=build_model,
    loss_fn=build_losses,
    optimizer_fn=lambda *_, **__: OPTIMIZER_V1,
    after_init=after_init
)
