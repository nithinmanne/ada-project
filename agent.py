"""This file contains the code for the agent implemented without any distributed learning
   It contains the main learning model, and has the model, environment and the state history
   stored inside it. It can be used by simply instantiating and calling train"""
import numpy as np
import gym

from model import create_dqn_model
from util.circular_buffer import CircularBuffer
from util.environment import MainPreprocessing, FrameStack
from config import *


class DQNAgent:
    """The main learning algorithm class, contains all the information
       and can be directly used to train"""
    def __init__(self):
        self.env = FrameStack(MainPreprocessing(gym.make(ENVIRONMENT)))
        self.action_space = self.env.action_space
        self.observation_shape = self.env.observation_space.shape
        self.observation_dtype = self.env.observation_space.dtype

        self.model = create_dqn_model(num_actions=self.action_space.n)
        self.model.compile(OPTIMIZER, LOSS_FUNCTION)
        self.target_model = create_dqn_model(num_actions=self.action_space.n)

        """Use the ring buffer to store the histories of the observations"""
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
        """Internal function that performs an action and saves it to the replay memory"""
        """Uses the epsilon-greedy strategy to generate action"""
        if self.epsilon > np.random.uniform():
            action = self.action_space.sample()
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        """Decrements the epsilon to slowly increase exploitation"""
        self.epsilon = max(self.epsilon - EPSILON_DECREMENT, EPSILON_MIN)

        state_next, reward, done, _ = self.env.step(action)
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)
        return state_next, reward, done

    def _learn(self):
        """Internal fucntion which is the main implementation of the learning algorithm"""
        sample_indices = np.random.choice(range(len(self.action_history)), size=BATCH_SIZE)
        action_sample = self.action_history[sample_indices]
        state_sample = self.state_history[sample_indices]
        state_next_sample = self.state_next_history[sample_indices]
        rewards_sample = self.rewards_history[sample_indices]
        done_sample = self.done_history[sample_indices]

        """This is the estimated Q value generated using the target model"""
        q_old = self.target_model.predict_on_batch(state_next_sample)
        batch_index = np.arange(BATCH_SIZE)
        if DDQN:
            """In Double DQN, instead of using the max action, we use the prediction
               model to generate the action that we will take"""
            q_eval = self.model.predict_on_batch(state_next_sample)
            max_actions = tf.argmax(q_eval, axis=1)
            selected_q_old = q_old[batch_index, max_actions]
        else:
            """In normal DQN, simply use the best action in this to train with"""
            selected_q_old = np.max(q_old, axis=1)

        """Finally update the target model with the estimate from the target model
           based on selection done above."""
        q_target = self.model.predict_on_batch(state_sample)
        q_target[batch_index, action_sample] = rewards_sample + (1 - done_sample) * GAMMA * selected_q_old
        return self.model.train_on_batch(state_sample, q_target)

    def train(self, num_episodes):
        """This is the main API of this class to start training. The input is
           the number of episodes, which the number of times the environment
           will reach the terminal state"""
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
        """This API is used to make the model play the game without any training or
           storing attached. This is meant to be executed on the final model to
           visually observe its performance. The model can also be loaded from file
           and used as it doesn't depend on any other variables generated while
           training."""
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
