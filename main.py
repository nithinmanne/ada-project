"""This is just a sample main file to call the non-distributed
   implementation of the agent, the agent is very easy to create
   and train without any config needed."""
import sys


import agent


# Windows CUDA Issue on my Laptop
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == '__main__':
    total_steps = 1000
    if len(sys.argv) == 2:
        total_steps = int(sys.argv[1])
    a = agent.DQNAgent()
    a.train(total_steps)
    a.model.save('model.h5')
    a.target_model.save('target_model.h5')
