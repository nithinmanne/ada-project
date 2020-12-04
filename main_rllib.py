"""This is just a sample main file to call the distributed
   implementation of the agent, the agent is very easy to create
   and train without any config needed. The agent can also be
   called using the Ray CLI as all the configuration is correctly
   defined in the default config."""
import sys
import ray


import trainer_rllib
from config import *


# Windows CUDA Issue on my Laptop
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == '__main__':
    total_steps = 1000
    if len(sys.argv) == 2:
        total_steps = int(sys.argv[1])
    ray.init()
    trainer = trainer_rllib.DQNRLlibTrainer(env=ENVIRONMENT, config={'num_workers': 1})
    for _ in range(total_steps):
        print(trainer.train())
