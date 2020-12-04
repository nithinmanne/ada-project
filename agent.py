import numpy as np
import gym
import tensorflow as tf

from model import create_dqn_model
from util.circular_buffer import CircularBuffer
from config import *


class DQNAgent:
    def __init__(self):
        self.env = gym.wrappers.FrameStack(
                        gym.wrappers.AtariPreprocessing(
                            gym.make(ENVIRONMENT),
                            scale_obs=True
                        ),
                        num_stack=4
        )
        self.action_space = self.env.action_space
        self.observation_shape = self.env.observation_space.shape
        self.observation_dtype = self.env.observation_space.dtype

        self.model = create_dqn_model(num_actions=self.action_space.n)
        self.model.compile(OPTIMIZER, LOSS_FUNCTION)
        self.target_model = create_dqn_model(num_actions=self.action_space.n)

        self.action_history = CircularBuffer(MAX_MEMORY_LENGTH, (), dtype=np.int8)
        self.state_history = CircularBuffer(MAX_MEMORY_LENGTH, self.observation_shape,
                                            dtype=self.observation_dtype)
        self.state_next_history = CircularBuffer(MAX_MEMORY_LENGTH, self.observation_shape,
                                                 dtype=self.observation_dtype)
        self.rewards_history = CircularBuffer(MAX_MEMORY_LENGTH, (), dtype=np.float32)
        self.done_history = CircularBuffer(MAX_MEMORY_LENGTH, (), dtype=int)

        self.episode_reward_history = CircularBuffer(40, (), dtype=int)
        self.epsilon = EPSILON_MAX

    def _action(self, state):
        if self.epsilon > np.random.uniform():
            action = self.action_space.sample()
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        self.epsilon = max(self.epsilon - EPSILON_DECREMENT, EPSILON_MIN)

        state_next, reward, done, _ = self.env.step(action)
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)
        return state_next, reward, done

    def _learn(self):
        sample_indices = np.random.choice(range(len(self.action_history)), size=BATCH_SIZE)
        action_sample = self.action_history[sample_indices]
        state_sample = self.state_history[sample_indices]
        state_next_sample = self.state_next_history[sample_indices]
        rewards_sample = self.rewards_history[sample_indices]
        done_sample = self.done_history[sample_indices]

        q_old = self.target_model.predict_on_batch(state_next_sample)
        batch_index = np.arange(BATCH_SIZE)
        if DQN:
            q_eval = self.model.predict_on_batch(state_next_sample)
            max_actions = tf.argmax(q_eval, axis=1)
            selected_q_old = q_old[batch_index, max_actions]
        else:
            selected_q_old = np.max(q_old, axis=1)

        q_target = self.model.predict_on_batch(state_sample)
        q_target[batch_index, action_sample] = rewards_sample + (1 - done_sample) * GAMMA * selected_q_old
        return self.model.train_on_batch(state_sample, q_target)

    def train(self, num_episodes):
        step = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            if RENDER:
                self.env.render()
            episode_reward = 0
            episode_loss = 0.

            for _ in range(MAX_EPISODE_STEPS):
                step += 1
                state, reward, done = self._action(state)
                if RENDER:
                    self.env.render()
                episode_reward += reward

                if step % MODEL_UPDATE_STEP_COUNT == 0 and step > BATCH_SIZE:
                    episode_loss += self._learn()
                if step % TARGET_MODEL_UPDATE_STEP_COUNT == 0 and step > BATCH_SIZE:
                    self.target_model.set_weights(self.model.get_weights())

                if done:
                    break

            self.episode_reward_history.append(episode_reward)
            print(f'Reward: {episode_reward}, Mean: {self.episode_reward_history.mean()}, \
Step: {step}, Loss: {episode_loss}')

    def play(self):
        state = self.env.reset()
        self.env.render()
        episode_reward = 0
        for _ in range(MAX_EPISODE_STEPS):
            if EPSILON_MIN > np.random.uniform():
                action = self.action_space.sample()
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.model(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()
            state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            self.env.render()
            if done:
                break
        print(f'Reward = {episode_reward}')
