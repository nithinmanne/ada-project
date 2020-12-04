"""This provides the pre-processing steps for the Gym environments. Gym provides an API
   to easily extend an existing environment, by just changing what we need, in this case,
   just the observation, without effecting any code for the actual environment."""
import cv2
import gym
import numpy as np

from util.circular_buffer import CircularBuffer

HEIGHT, WIDTH = 84, 84


class MainPreprocessing(gym.ObservationWrapper):
    """This does the main pre-processing for the environment.
       It does:
       1. Convert Image to Grayscale
       2. Resize the Image to WIDTHxHEIGHT
       3. Scale the values by 255 to be in 0-1 range."""
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
    """This is the Frame Stacking API which stores the last FRAME_STACK_COUNT
       frames and returns them all to be used for training."""
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
        """Its necessary to in fact override reset as well since we
           don't want to include frames from previous completely
           unrelated runs along with the first observation."""
        obs = self.env.reset()
        for _ in range(self.frame_count):
            self.frames.append(obs)
        return self.get_frames()

    def observation(self, obs):
        self.frames.append(obs)
        return self.get_frames()
