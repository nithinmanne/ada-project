import cv2
import gym
import numpy as np

from util.circular_buffer import CircularBuffer

HEIGHT, WIDTH = 84, 84


class MainPreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (WIDTH, HEIGHT)
        self.observation_space = gym.spaces.Box(
            low=0.,
            high=1.,
            shape=self.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return np.array(obs).astype(np.float32)/255


FRAME_STACK_COUNT = 4


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frame_count = FRAME_STACK_COUNT
        self.frames = CircularBuffer(4, env.observation_space.shape)
        self.observation_space = gym.spaces.Box(
            low=np.array([env.observation_space.low]*4),
            high=np.array([env.observation_space.high]*4),
            shape=(self.frame_count, *env.observation_space.shape),
            dtype=env.observation_space.dtype
        )

    def get_frames(self):
        return self.frames[np.arange(self.frame_count)]

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.frame_count):
            self.frames.append(obs)
        return self.get_frames()

    def observation(self, obs):
        self.frames.append(obs)
        return self.get_frames()
