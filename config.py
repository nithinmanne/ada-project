"""This file as the hyper-parameters and configurations for the agent. I put it all in one file rather
   than part of constructors, as I wanted to use the exact same config in both implementations, and
   this was the most convenient way."""
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import Huber
from ray.rllib.agents.trainer import with_common_config


"""This is the Gym environment that I used"""
ENVIRONMENT = 'BreakoutNoFrameskip-v4'
"""This flag is used to represent if the user wants to render the actual screen while training"""
RENDER = True

"""This is the discount value of future reward, its pretty high as the main goal is to
   optimize for overall reward anyway but not spend too much time doing nothing"""
GAMMA = 0.99
"""This is the final value of epsilon in the epsilon greedy policy which has a decaying epsilon"""
EPSILON_MIN = 0.1
"""This is the initial value of epsilon. Its value is initially grater than 1 as that lets
   the agent perform only exploration for the initial time when its greater than 1 without
   any other extra logic to ensure that"""
EPSILON_MAX = 1.1
"""This is the total number of frames that the value of epsilon decays for"""
EPSILON_DECREMENT_FRAMES = 10_000
"""This is just the above difference calculated so it can be used directly in code"""
EPSILON_DECREMENT = (EPSILON_MAX - EPSILON_MIN) / EPSILON_DECREMENT_FRAMES
"""This is the batch size that is used to train the neural network with"""
BATCH_SIZE = 32
"""This is the maximum length that any one episode is allowed to run before stopping.
   This ensures that the environment doesn't somehow get stuck in an infinite state"""
MAX_EPISODE_STEPS = 1_000
"""This the size of the replay buffer used"""
MAX_MEMORY_LENGTH = 100_000
"""This is the number of steps after which the model is trained, this number is quite small
   as it lets the model learn much faster"""
MODEL_UPDATE_STEP_COUNT = 4
"""This is the number of steps after which the target network updates with weights from
   the main network, this number is relatively large to ensure a stable learning process"""
TARGET_MODEL_UPDATE_STEP_COUNT = 1000

"""This is a flag to select whether to use Double DQN or not"""
DDQN = True
"""This is the optimizer that I used for the neural network"""
OPTIMIZER = Adam(learning_rate=0.00025, clipnorm=1.0)
"""This is the same optimizer as above, but latest version of RLlib
   seems to expect the optimizer to have the same API as
   TensorFlow v1 even though most of the library supports
   TensorFlow v2. I only found this by looking at the source code
   where it calls the optimizer function using the old API.
   But luckily since the TensorFlow APIs are
   completely backwards compatible, I can use this function as a
   drop-in replacement on the same model defined in TensorFlow v2,
   and it works."""
OPTIMIZER_V1 = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00025)
"""This is just the loss function that I used for training the network"""
LOSS_FUNCTION = Huber()

"""These are mostly the same config as above, just in the format that RLlib expects it as"""
RLLIB_DEFAULT_CONFIG = with_common_config({
    'exploration_config': {
        'type': 'EpsilonGreedy',
        'initial_epsilon': EPSILON_MAX,
        'final_epsilon': EPSILON_MIN,
        'epsilon_timesteps': EPSILON_DECREMENT_FRAMES,
    },
    'evaluation_config': {
        'explore': False,
    },
    'timesteps_per_iteration': MAX_EPISODE_STEPS,
    'target_network_update_freq': TARGET_MODEL_UPDATE_STEP_COUNT,
    'buffer_size': MAX_MEMORY_LENGTH,
    'compress_observations': False,
    'before_learn_on_batch': None,
    'training_intensity': None,

    'learning_starts': 0,  # Handled by an epsilon value of greater than 1
    'rollout_fragment_length': MODEL_UPDATE_STEP_COUNT,
    'train_batch_size': BATCH_SIZE,

    'num_workers': 0,  # Can be modified here or while calling in the CLI
    'num_gpus': 0,  # Can be modified here or while calling in the CLI
    'worker_side_prioritization': False,
    'min_iter_time_s': 1,
    '_use_trajectory_view_api': True,
})

"""I used the word 'this' 26 times including the 'this'es in this line"""
