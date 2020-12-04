from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import Huber
from ray.rllib.agents.trainer import with_common_config


ENVIRONMENT = 'BreakoutNoFrameskip-v4'
RENDER = True

GAMMA = 0.99
EPSILON_MIN = 0.1
EPSILON_MAX = 1.1
EPSILON_DECREMENT_FRAMES = 10_000
EPSILON_DECREMENT = (EPSILON_MAX - EPSILON_MIN) / EPSILON_DECREMENT_FRAMES
BATCH_SIZE = 32
MAX_EPISODE_STEPS = 1_000
MAX_MEMORY_LENGTH = 2_000
MODEL_UPDATE_STEP_COUNT = 4
TARGET_MODEL_UPDATE_STEP_COUNT = 4

DQN = True
OPTIMIZER = Adam(learning_rate=0.00025, clipnorm=1.0)
OPTIMIZER_V1 = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00025)
LOSS_FUNCTION = Huber()

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

    'learning_starts': 0,  # int((EPSILON_MAX - 1) / EPSILON_DECREMENT),
    'rollout_fragment_length': MODEL_UPDATE_STEP_COUNT,
    'train_batch_size': BATCH_SIZE,

    'num_workers': 0,
    'num_gpus': 0,
    'worker_side_prioritization': False,
    'min_iter_time_s': 1,
    '_use_trajectory_view_api': True,
})
